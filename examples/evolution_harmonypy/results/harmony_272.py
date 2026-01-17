# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import logging

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_device(device=None):
    """Get the appropriate device for PyTorch operations."""
    if device is not None:
        return torch.device(device)

    # Check for available accelerators
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta=None,
    lamb=None,
    sigma=0.1,
    nclust=None,
    tau=0,
    block_size=0.05,
    max_iter_harmony=10,
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4,
    alpha=0.2,
    verbose=True,
    random_state=0,
    device=None
):
    """Run Harmony batch effect correction.

    This is a PyTorch implementation matching the R package formulas.
    Supports CPU and GPU (CUDA, MPS) acceleration.

    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables (cells x variables)
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is 1 for each batch.
        If -1, lambda is estimated automatically (matches R package).
    sigma : float, optional
        Kernel bandwidth for soft clustering. Default is 0.1.
    nclust : int, optional
        Number of clusters. Default is min(N/30, 100).
    tau : float, optional
        Protection against overcorrection. Default is 0.
    block_size : float, optional
        Proportion of cells to update in each block. Default is 0.05.
    max_iter_harmony : int, optional
        Maximum Harmony iterations. Default is 10.
    max_iter_kmeans : int, optional
        Maximum k-means iterations per Harmony iteration. Default is 20.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-5.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-4.
    alpha : float, optional
        Alpha parameter for lambda estimation (when lamb=-1). Default is 0.2.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps'). Default is auto-detect.

    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells"

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    if isinstance(sigma, float) and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Create batch indicator matrix (one-hot encoded)
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T.astype(np.float32)
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float32)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float32)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R package)
    lambda_estimation = False
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif lamb == -1:
        lambda_estimation = True
        lamb = np.zeros(1, dtype=np.float32)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    else:
        lamb = np.asarray(lamb, dtype=np.float32)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float32)

    # Number of items in each category
    N_b = phi.sum(axis=1)
    Pr_b = (N_b / N).astype(np.float32)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    # Get device
    device_obj = get_device(device)

    if verbose:
        logger.info(f"Running Harmony (PyTorch on {device_obj})")
        logger.info("  Parameters:")
        logger.info(f"    max_iter_harmony: {max_iter_harmony}")
        logger.info(f"    max_iter_kmeans: {max_iter_kmeans}")
        logger.info(f"    epsilon_cluster: {epsilon_cluster}")
        logger.info(f"    epsilon_harmony: {epsilon_harmony}")
        logger.info(f"    nclust: {nclust}")
        logger.info(f"    block_size: {block_size}")
        if lambda_estimation:
            logger.info(f"    lamb: dynamic (alpha={alpha})")
        else:
            logger.info(f"    lamb: {lamb[1:]}")
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs 脳 {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values
    data_mat = np.asarray(data_mat, dtype=np.float32)

    ho = Harmony(
        data_mat, phi, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state, device_obj
    )

    return ho


class Harmony:
    """Harmony class for batch effect correction using PyTorch.

    Supports CPU and GPU acceleration.
    """

    def __init__(
            self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
            max_iter_harmony, max_iter_kmeans,
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state, device
    ):
        self.device = device

        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)

        # Simple L2 normalization
        self._Z_cos = F.normalize(self._Z_orig, p=2, dim=0, eps=1e-12)

        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]

        # Average batch size heuristic (used to choose vectorization strategy)
        self.avg_batch_size = float(self.N) / float(max(self.B, 1))

        # ---------------------------------------------------------------------
        # Precompute compact batch_id and contiguous batch blocks (for fast grouping)
        # ---------------------------------------------------------------------
        self._batch_id = torch.argmax(self._Phi, dim=0).to(torch.int64)  # (N,)
        self._perm = torch.argsort(self._batch_id)  # group cells by batch
        self._inv_perm = torch.empty_like(self._perm)
        self._inv_perm[self._perm] = torch.arange(self.N, device=self.device, dtype=self._perm.dtype)

        # batch_ptr: (B+1,) pointers into perm such that batch b lives in [ptr[b], ptr[b+1])
        batch_counts = torch.bincount(self._batch_id, minlength=self.B)
        self._batch_ptr = torch.zeros(self.B + 1, device=self.device, dtype=torch.int64)
        self._batch_ptr[1:] = torch.cumsum(batch_counts, dim=0)

        # Build batch index for legacy code paths / occasional indexing
        self._batch_index = []
        for b in range(self.B):
            idx = torch.where(self._Phi[b, :] > 0)[0]
            self._batch_index.append(idx)

        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self._lamb = torch.tensor(lamb, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self._sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self._theta = torch.tensor(theta, dtype=torch.float32, device=device)

        # ---------------------------------------------------------------------
        # Assignment / correction stabilization hyperparameters (internal defaults)
        # ---------------------------------------------------------------------
        # IPF batch-balancing strength schedule (rho): high early, lower later.
        self.rho_start = 0.6
        self.rho_end = 0.2
        self.ipf_iters = 5
        self.ipf_eps = 1e-8

        # Trust-region / step-size for MOE correction
        self.eta_init = 1.0
        self.eta_min = 0.125
        self.eta_max_backtracks = 4
        self.r_stability_thresh = 0.10  # mean absolute change in R allowed for acceptance

        # Track harmony iteration for rho schedule
        self._harmony_iter = 0

        # Tie-to-mean regularizer strength for MOE coefficients (stability / bio conservation)
        self.gamma_w = 1e-2

        # Dirty flags to avoid redundant distance recomputation
        self._dist_dirty = True

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================

    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        return self._Z_corr.cpu().numpy().T

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        return self._Z_orig.cpu().numpy().T

    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        return self._Z_cos.cpu().numpy().T

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        return self._R.cpu().numpy().T

    @property
    def Y(self):
        """Cluster centroids matrix (d x K). Columns are cluster centers."""
        return self._Y.cpu().numpy()

    @property
    def O(self):
        """Observed batch-cluster counts (K x B). O[k,b] = sum of R[k,:] for batch b."""
        return self._O.cpu().numpy()

    @property
    def E(self):
        """Expected batch-cluster counts (K x B). E[k,b] = cluster_size[k] * batch_proportion[b]."""
        return self._E.cpu().numpy()

    @property
    def Phi(self):
        """Batch indicator matrix (N x B). One-hot encoding of batch membership."""
        return self._Phi.cpu().numpy().T

    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        return self._Phi_moe.cpu().numpy().T

    @property
    def Pr_b(self):
        """Batch proportions (B,). Pr_b[b] = cells in batch b / total cells."""
        return self._Pr_b.cpu().numpy()

    @property
    def theta(self):
        """Diversity penalty parameters (B,). Higher = more mixing encouraged."""
        return self._theta.cpu().numpy()

    @property
    def sigma(self):
        """Clustering bandwidth parameters (K,). Soft assignment kernel width."""
        return self._sigma.cpu().numpy()

    @property
    def lamb(self):
        """Ridge regression penalty ((B+1),). Regularization for batch correction."""
        return self._lamb.cpu().numpy()

    @property
    def objectives(self):
        """List of objective values for compatibility with evaluator."""
        return self.objective_harmony

    def result(self):
        """Return corrected data as NumPy array."""
        return self._Z_corr.cpu().numpy().T

    def allocate_buffers(self):
        self._scale_dist = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._dist_mat = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._W = torch.zeros((self.B + 1, self.d), dtype=torch.float32, device=self.device)
        self._R = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._Y = torch.zeros((self.d, self.K), dtype=torch.float32, device=self.device)

        # Buffers to reduce allocations in update_R()
        self._O_buf = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E_buf = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._G_buf = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._T_buf = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._S_buf = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._R_sum_buf = torch.zeros((1, self.N), dtype=torch.float32, device=self.device)

        # Track previous responsibilities across calls (for adaptive IPF)
        self._R_prev = None

    def init_cluster(self, random_state):
        logger.info("Computing initial centroids with sklearn.KMeans...")
        # KMeans needs CPU numpy array
        Z_cos_np = self._Z_cos.cpu().numpy()
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(Z_cos_np.T)
        self._Y = torch.tensor(model.cluster_centers_.T, dtype=torch.float32, device=self.device)
        logger.info("KMeans initialization complete.")

        # Normalize centroids
        self._Y = F.normalize(self._Y, p=2, dim=0, eps=1e-12)

        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

        # Compute R
        self._R = -self._dist_mat / self._sigma[:, None]
        self._R = torch.exp(self._R)
        self._R = self._R / self._R.sum(dim=0)

        # Batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective_value(self):
        """Compute (but do not record) the kmeans objective value and its components.

        Returns
        -------
        tuple
            (objective, kmeans_error, entropy, cross_entropy) all normalized.
        """
        # Normalization constant
        norm_const = 2000.0 / self.N

        # K-means error
        kmeans_error = torch.sum(self._R * self._dist_mat)

        # Entropy
        _entropy = torch.sum(safe_entropy_torch(self._R) * self._sigma[:, None])

        # Cross entropy (R package formula) with numerical stability
        R_sigma = self._R * self._sigma[:, None]
        # Clamp to avoid log(0) or division by zero
        O_clamped = torch.clamp(self._O, min=1e-8)
        E_clamped = torch.clamp(self._E, min=1e-8)

        # log((O+E)/E) = log1p(O/E) is more stable than forming ratio explicitly
        O_over_E = O_clamped / E_clamped
        theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log1p(O_over_E)

        _cross_entropy = torch.sum(R_sigma * (theta_log @ self._Phi))

        # Normalize
        kmeans_error_n = (kmeans_error * norm_const).item()
        entropy_n = (_entropy * norm_const).item()
        cross_entropy_n = (_cross_entropy * norm_const).item()
        objective_n = kmeans_error_n + entropy_n + cross_entropy_n

        return objective_n, kmeans_error_n, entropy_n, cross_entropy_n

    def compute_objective(self):
        """Compute and record objective values (backwards compatible)."""
        objective_n, kmeans_error_n, entropy_n, cross_entropy_n = self.compute_objective_value()
        self.objective_kmeans.append(objective_n)
        self.objective_kmeans_dist.append(kmeans_error_n)
        self.objective_kmeans_entropy.append(entropy_n)
        self.objective_kmeans_cross.append(cross_entropy_n)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            self._harmony_iter = i
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")

            self.cluster()
            self.moe_correct_ridge()

            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break

        if verbose and not converged:
            logger.info("Stopped before convergence")

    def cluster(self):
        # Alternating updates:
        #  1) update centroids Y from current assignments R
        #  2) update assignments R using batch-balanced entropic OT (Sinkhorn)
        # This makes batch mixing an explicit constraint rather than a heuristic penalty.

        if self._dist_dirty:
            self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)
            self._dist_dirty = False

        rounds = 0
        obj_every = 2

        for i in range(self.max_iter_kmeans):
            # Update Y
            self._Y = self._Z_cos @ self._R.T
            self._Y = F.normalize(self._Y, p=2, dim=0, eps=1e-12)
            self._dist_dirty = True

            # Update distance matrix
            if self._dist_dirty:
                self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)
                self._dist_dirty = False

            # Update R via balanced OT
            self.update_R()

            # Compute objective less frequently; convergence check only on objective steps
            if (i % obj_every) == 0:
                self.compute_objective()
                if i > self.window_size:
                    if self.check_convergence(0):
                        rounds = i + 1
                        break

            rounds = i + 1

        # Ensure we have an objective recorded for this cluster() call
        if len(self.objective_kmeans) == 0 or len(self.objective_harmony) == 0:
            self.compute_objective()

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        """Update soft assignments R using batch-constrained IPF on responsibilities.

        Steps:
          1) Compute base responsibilities from distances only using a stable softmax.
          2) Enforce batch balancing by multiplicative reweighting using IPF, targeting:
                T = (1-rho)*O + rho*E
             where O is current observed batch-cluster mass and E is expected mass.
          3) Renormalize per cell after each IPF sweep.
        """
        # ------------------------------------------------------------------
        # 1) Base responsibilities from distances only (stable softmax)
        # ------------------------------------------------------------------
        logits = -self._dist_mat / self._sigma[:, None]  # K x N
        self._R = torch.softmax(logits, dim=0)

        # ------------------------------------------------------------------
        # 2) IPF to (softly) enforce batch composition per cluster (vectorized)
        # ------------------------------------------------------------------
        # Schedule rho (mixing strength): high early, lower later
        if self.max_iter_harmony > 1:
            t = float(max(self._harmony_iter - 1, 0)) / float(self.max_iter_harmony - 1)
        else:
            t = 0.0
        rho = (1.0 - t) * self.rho_start + t * self.rho_end

        # Adaptive IPF sweeps based on assignment stability across calls
        ipf_iters = int(self.ipf_iters)
        if self._R_prev is not None:
            # Use float16 prev copy (when available) to cut bandwidth
            prev = self._R_prev.to(dtype=self._R.dtype)
            delta = torch.mean(torch.abs(self._R - prev)).item()
            if delta > 0.15:
                ipf_iters = max(2, min(ipf_iters, 3))
            elif delta < 0.05:
                ipf_iters = min(7, max(ipf_iters, 5))

        prev_dtype = torch.float16 if self._R.is_cuda or (self._R.device.type == 'mps') else torch.float32
        self._R_prev = self._R.detach().to(dtype=prev_dtype)

        # Reuse buffers to reduce allocations inside the loop (fully in-place)
        for _ in range(ipf_iters):
            # Current O and E from current R (fill buffers in-place)
            torch.matmul(self._R, self._Phi.T, out=self._O_buf)  # K x B
            self._E_buf.copy_(torch.outer(self._R.sum(dim=1), self._Pr_b))  # K x B

            # T = (1-rho)*O + rho*E into buffer
            self._T_buf.copy_(self._O_buf).mul_(1.0 - rho).add_(self._E_buf, alpha=rho)

            # S = T / O into buffer
            self._S_buf.copy_(self._T_buf).div_(torch.clamp(self._O_buf, min=self.ipf_eps))

            # Convert to per-cell scaling using one-hot Phi: (K x B) @ (B x N) -> (K x N)
            torch.matmul(self._S_buf, self._Phi, out=self._G_buf)  # K x N

            # Apply scaling in-place and renormalize per cell in-place
            self._R.mul_(self._G_buf)
            self._R_sum_buf.copy_(self._R.sum(dim=0, keepdim=True))
            self._R.div_(torch.clamp(self._R_sum_buf, min=self.ipf_eps))

        # ------------------------------------------------------------------
        # 3) Update batch diversity statistics after final IPF update
        # ------------------------------------------------------------------
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False

            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            denom = abs(obj_old) + 1e-12
            return abs(obj_old - obj_new) / denom < self.epsilon_kmeans

        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False

            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            denom = abs(obj_old) + 1e-12
            return (obj_old - obj_new) / denom < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects with trust-region step size.

        Computes the usual MOE ridge correction, but applies it with an adaptive
        step-size eta using a short backtracking line search. Acceptance requires:
          - objective improvement (kmeans objective proxy decreases)
          - responsibilities remain stable (mean |R_new - R_old| not too large)

        This prevents overcorrection when R is noisy/unstable early on.
        """
        with torch.no_grad():
            # Precompute some shared quantities
            Z = self._Z_orig     # d x N
            Z_T = Z.T.contiguous()  # N x d

            # ------------------------------------------------------------------
            # Compute sufficient statistics exploiting one-hot Phi_moe structure
            # Avoids materializing (K,B,N) and (K,d,N) intermediates.
            # ------------------------------------------------------------------
            # r_sum: (K,)
            r_sum = self._R.sum(dim=1)

            # O: (K,B) already maintained by update_R(); treat as r_kb
            O_kb = self._O  # K x B

            # z_sum: (K,d) = sum_i R[k,i] * Z[:,i]
            z_sum_kd = (Z @ self._R.T).T  # K x d

            # z_kb: (K,B,d) = sum_{i in batch b} R[k,i] * Z[:,i]
            # Heuristic: if many batches / small batches -> index_add; else per-batch matmul
            use_index_add = (self.B >= 32) or (self.avg_batch_size <= 256.0)
            z_kb_kbd = torch.zeros((self.K, self.B, self.d), dtype=torch.float32, device=self.device)

            if use_index_add:
                # Loop over clusters (K usually small), scatter-add into batch bins
                for k in range(self.K):
                    w = self._R[k, :]  # (N,)
                    weighted = Z_T * w[:, None]  # (N,d)
                    z_kb_kbd[k].index_add_(0, self._batch_id, weighted)
            else:
                # Use contiguous per-batch blocks based on permutation (fewer gathers)
                perm = self._perm
                ptr = self._batch_ptr
                R_sorted = self._R[:, perm]   # K x N (grouped by batch)
                Z_sorted = Z[:, perm]         # d x N (grouped by batch)

                for b in range(self.B):
                    s = int(ptr[b].item())
                    e = int(ptr[b + 1].item())
                    if e <= s:
                        continue
                    Rb = R_sorted[:, s:e]        # K x n_b (contiguous)
                    Zb = Z_sorted[:, s:e]        # d x n_b (contiguous)
                    z_kb_kbd[:, b, :] = (Rb @ Zb.T)  # K x d

            # ------------------------------------------------------------------
            # Assemble ridge systems A and RHS S_phi_z using block structure
            # ------------------------------------------------------------------
            # Lambda (K, B+1)
            if self.lambda_estimation:
                lamb_k = torch.zeros((self.K, self.B + 1), dtype=torch.float32, device=self.device)
                lamb_k[:, 1:] = self._E * self.alpha
            else:
                lamb_k = self._lamb.unsqueeze(0).expand(self.K, -1)

            # A: (K, B+1, B+1)
            A = torch.zeros((self.K, self.B + 1, self.B + 1), dtype=torch.float32, device=self.device)
            A[:, 0, 0] = r_sum
            A[:, 0, 1:] = O_kb
            A[:, 1:, 0] = O_kb
            A[:, 1:, 1:] = torch.diag_embed(O_kb)

            # Add diagonal ridge penalties
            A = A + torch.diag_embed(lamb_k)

            # RHS: S_phi_z: (K, B+1, d)
            S_phi_z = torch.zeros((self.K, self.B + 1, self.d), dtype=torch.float32, device=self.device)
            S_phi_z[:, 0, :] = z_sum_kd
            S_phi_z[:, 1:, :] = z_kb_kbd

            # Solve for W_all: (K, B+1, d)
            L = torch.linalg.cholesky(A)
            W_all = torch.cholesky_solve(S_phi_z, L)
            W_all[:, 0, :] = 0  # Do not remove intercept

            # Tie-to-mean regularizer (shrink cluster-specific W_k toward global mean)
            if self.gamma_w is not None and self.gamma_w > 0:
                W_mean = W_all.mean(dim=0, keepdim=True)  # 1 x (B+1) x d
                W_all = (W_all + self.gamma_w * W_mean) / (1.0 + self.gamma_w)
                W_all[:, 0, :] = 0  # preserve intercept rule after shrinkage

            # ------------------------------------------------------------------
            # Compute correction using grouped batches (contiguous blocks)
            # correction[:, idx] = Wb.T @ Rb for cells in batch b, but done on grouped order
            # ------------------------------------------------------------------
            perm = self._perm
            inv_perm = self._inv_perm
            ptr = self._batch_ptr

            R_sorted = self._R[:, perm]  # K x N (contiguous by batch)
            correction_dn_sorted = torch.zeros((self.d, self.N), dtype=torch.float32, device=self.device)

            for b in range(self.B):
                s = int(ptr[b].item())
                e = int(ptr[b + 1].item())
                if e <= s:
                    continue
                Rb = R_sorted[:, s:e]           # K x n_b
                Wb = W_all[:, 1 + b, :]         # K x d
                correction_dn_sorted[:, s:e] = Wb.T @ Rb

            correction_dn = correction_dn_sorted[:, inv_perm]

            # ------------------------------------------------------------------
            # Trust-region / backtracking without mutating global state during trials
            # ------------------------------------------------------------------
            obj_old = self.objective_kmeans[-1] if len(self.objective_kmeans) > 0 else float('inf')
            R_old = self._R  # do not clone big tensors; treat as read-only in trials

            accepted = False
            eta = float(self.eta_init)

            # Lightweight candidate R update (softmax + few IPF iters) with no side effects
            def _candidate_R_from_dist(dist_mat, ipf_iters_eval=2):
                # Base responsibilities
                logits = -dist_mat / self._sigma[:, None]
                R_cand = torch.softmax(logits, dim=0)

                if ipf_iters_eval <= 0:
                    # Still need O/E for objective
                    O_cand = R_cand @ self._Phi.T
                    E_cand = torch.outer(R_cand.sum(dim=1), self._Pr_b)
                    return R_cand, O_cand, E_cand

                # rho schedule consistent with main update
                if self.max_iter_harmony > 1:
                    t_loc = float(max(self._harmony_iter - 1, 0)) / float(self.max_iter_harmony - 1)
                else:
                    t_loc = 0.0
                rho_loc = (1.0 - t_loc) * self.rho_start + t_loc * self.rho_end

                for _ in range(ipf_iters_eval):
                    O_cand = R_cand @ self._Phi.T
                    E_cand = torch.outer(R_cand.sum(dim=1), self._Pr_b)
                    T_cand = (1.0 - rho_loc) * O_cand + rho_loc * E_cand
                    S_cand = T_cand / torch.clamp(O_cand, min=self.ipf_eps)
                    G_cand = S_cand @ self._Phi
                    R_cand = R_cand * G_cand
                    R_cand = R_cand / torch.clamp(R_cand.sum(dim=0, keepdim=True), min=self.ipf_eps)

                O_cand = R_cand @ self._Phi.T
                E_cand = torch.outer(R_cand.sum(dim=1), self._Pr_b)
                return R_cand, O_cand, E_cand

            for _ in range(self.eta_max_backtracks + 1):
                # Candidate corrected embedding
                Z_corr_candidate = self._Z_orig - (eta * correction_dn)
                Z_cos_candidate = F.normalize(Z_corr_candidate, p=2, dim=0, eps=1e-12)
                dist_candidate = 2 * (1 - self._Y.T @ Z_cos_candidate)

                R_cand, O_cand, E_cand = _candidate_R_from_dist(dist_candidate, ipf_iters_eval=2)

                # Compute objective for candidate without mutating state
                norm_const = 2000.0 / self.N
                kmeans_error = torch.sum(R_cand * dist_candidate)
                entropy = torch.sum(safe_entropy_torch(R_cand) * self._sigma[:, None])

                O_clamped = torch.clamp(O_cand, min=1e-8)
                E_clamped = torch.clamp(E_cand, min=1e-8)
                O_over_E = O_clamped / E_clamped
                theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log1p(O_over_E)
                cross_entropy = torch.sum((R_cand * self._sigma[:, None]) * (theta_log @ self._Phi))

                obj_new = (kmeans_error + entropy + cross_entropy) * norm_const
                obj_new = obj_new.item()

                r_change = torch.mean(torch.abs(R_cand - R_old)).item()

                if (obj_new <= obj_old) and (r_change <= self.r_stability_thresh):
                    accepted = True
                    # Commit state once
                    self._Z_corr = Z_corr_candidate
                    self._Z_cos = Z_cos_candidate
                    self._dist_mat = dist_candidate
                    self._R = R_cand
                    self._O = O_cand
                    self._E = E_cand
                    self._dist_dirty = False
                    self.compute_objective()
                    break

                eta *= 0.5
                if eta < self.eta_min:
                    break

            if not accepted:
                # Fall back to a conservative small step (no line-search side-effects)
                eta = max(float(self.eta_min), min(float(self.eta_init), 1.0))
                self._Z_corr = self._Z_orig - (eta * correction_dn)
                self._Z_cos = F.normalize(self._Z_corr, p=2, dim=0, eps=1e-12)
                self._dist_dirty = True
                # Keep R/O/E as-is (from previous clustering step) to avoid destabilizing loops
            # else: committed & objective already appended


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * torch.log(x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column."""
    result = torch.empty_like(A)
    for c in range(A.shape[1]):
        result[:, c] = torch.pow(A[:, c], T[c])
    return result


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
