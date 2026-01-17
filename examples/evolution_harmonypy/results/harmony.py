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

        # Simple L2 normalization (clamp to avoid NaN/Inf for near-zero columns)
        _znorm = torch.linalg.norm(self._Z_orig, ord=2, dim=0).clamp_min(1e-12)
        self._Z_cos = self._Z_orig / _znorm

        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]

        # Build batch index for fast ridge correction
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
        # OT/Sinkhorn assignment hyperparameters (internal defaults)
        # These replace the previous blockwise heuristic update_R.
        # ---------------------------------------------------------------------
        self.ot_eps = float(torch.median(self._sigma).item())  # entropic regularization
        self.ot_max_iter = 50
        self.ot_tol = 1e-3

        # Deterministic annealing for assignment temperature (smooth -> sharp)
        # Applied as a multiplier on ot_eps inside update_R().
        self.anneal_T_start = 2.0
        self.anneal_T_end = 1.0

        # Convergence based on assignment stability (instead of noisy objective windowing)
        self.r_converge_tol = 1e-3

        # Tie-to-mean regularizer strength for MOE coefficients (stability / bio conservation)
        self.gamma_w = 1e-2

        # Cache broadcastable theta row for update_R (1 x B)
        self._theta_row = self._theta.view(1, -1)

        # ---------------------------------------------------------------------
        # Caches / precomputations for speed & stability
        # ---------------------------------------------------------------------
        # Per-cell batch id (Phi is one-hot across batches)
        self._batch_id = torch.argmax(self._Phi, dim=0)

        # Precompute inverse sigma (sigma is fixed after init)
        self._inv_sigma = 1.0 / torch.clamp(self._sigma, min=1e-12)

        # Per-Nb caches to avoid repeated allocations in update_R
        self._ones_cache = {}  # Nb -> ones(Nb)
        self._v_cache = {}     # Nb -> ones(Nb) (used as initial v)

        # Track assignment stability for Harmony-level convergence
        self._last_delta_R = None
        self._harmony_stable_steps = 0
        self.harmony_delta_tol = 5e-4  # internal stopping threshold on delta_R

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
        self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)

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

    def compute_objective(self):
        # Normalization constant
        norm_const = 2000.0 / self.N

        # K-means error
        kmeans_error = torch.sum(self._R * self._dist_mat).item()

        # Entropy
        _entropy = torch.sum(safe_entropy_torch(self._R) * self._sigma[:, None]).item()

        # Cross entropy (R package formula) with numerical stability.
        # Exploit one-hot Phi: (theta_log @ Phi) == theta_log[:, batch_id]
        R_sigma = self._R * self._sigma[:, None]
        O_clamped = torch.clamp(self._O, min=1e-8)
        E_clamped = torch.clamp(self._E, min=1e-8)

        # Stable log1p form: log(1 + O/E)
        theta_log = self._theta_row * torch.log1p(O_clamped / E_clamped)  # (1,B) * (K,B) -> (K,B)
        batch_term = theta_log[:, self._batch_id]  # (K, N)
        _cross_entropy = torch.sum(R_sigma * batch_term).item()

        # Store with normalization constant
        self.objective_kmeans.append((kmeans_error + _entropy + _cross_entropy) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_cross_entropy * norm_const)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")

            # No gradients needed anywhere in Harmony
            with torch.no_grad():
                self.cluster()
                self.moe_correct_ridge()

            # Prefer assignment-stability based stopping (objective can be noisy)
            if self._last_delta_R is not None and self._last_delta_R < self.harmony_delta_tol:
                self._harmony_stable_steps += 1
            else:
                self._harmony_stable_steps = 0

            # Require stability for 2 consecutive Harmony steps
            if self._harmony_stable_steps >= 2:
                converged = True
            else:
                # Fallback to objective-based check
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
        #  2) update assignments R using batch-marginal-constrained entropic OT (Sinkhorn)
        # This makes batch mixing an explicit constraint rather than a heuristic penalty.

        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

        rounds = 0
        delta_R = None
        for i in range(self.max_iter_kmeans):
            # Update Y
            self._Y = self._Z_cos @ self._R.T
            self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)

            # Update distance matrix
            self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

            # Deterministic annealing (smooth -> sharp assignments across kmeans rounds)
            if self.max_iter_kmeans > 1:
                frac = float(i) / float(self.max_iter_kmeans - 1)
            else:
                frac = 1.0
            T = self.anneal_T_start + (self.anneal_T_end - self.anneal_T_start) * frac

            # Update R via balanced OT, and use assignment stability for convergence
            delta_R = self.update_R(T=T)

            # Objective is diagnostic only (and can be noisy); compute sparsely.
            should_compute = (i <= self.window_size + 1) or (i % 2 == 0) or (i == self.max_iter_kmeans - 1)
            if should_compute:
                self.compute_objective()

            # Converge based on assignment stability (primary signal)
            if delta_R is not None and delta_R < self.r_converge_tol:
                rounds = i + 1
                break

            rounds = i + 1

        # Save the last assignment delta for Harmony-level convergence checks
        self._last_delta_R = delta_R

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self, T=1.0):
        """Update soft assignments R using per-batch marginal-constrained entropic OT (Sinkhorn).

        For each batch b, solve a KL-projected assignment on the submatrix (K x N_b):

            minimize   <C, R> + eps * KL(R || 1)
            s.t.       sum_k R[k,n] = 1              (each cell is a simplex over clusters)
                       sum_n R[k,n] = pi_k * N_b     (cluster marginals within the batch)

        Where:
            C[k,n] = dist(k,n) / sigma_k
            pi_k   = global cluster weights (current) or uniform fallback

        Deterministic annealing is applied by scaling eps <- eps * T (T decreases toward 1).
        Returns:
            delta_R : float
                Mean absolute change in assignments (for convergence checks).
        """
        # Global cluster proportions (pi). Use current R; fallback to uniform if degenerate.
        pi = self._R.sum(dim=1)
        pi = pi / torch.clamp(pi.sum(), min=1e-12)
        if not torch.isfinite(pi).all() or float(pi.min().item()) <= 0.0:
            pi = torch.full((self.K,), 1.0 / self.K, dtype=torch.float32, device=self.device)

        # Annealed entropic regularization
        eps = float(self.ot_eps) * float(T)
        eps = max(eps, 1e-8)

        # Accumulate assignment change without cloning the full KxN matrix
        delta_R = 0.0

        # Update each batch independently
        for idx in self._batch_index:
            Nb = idx.numel()
            if Nb == 0:
                continue

            # Cache per-Nb vectors to avoid repeated allocations
            b = self._ones_cache.get(Nb)
            if b is None or b.device != self.device:
                b = torch.ones((Nb,), dtype=torch.float32, device=self.device)
                self._ones_cache[Nb] = b

            v0 = self._v_cache.get(Nb)
            if v0 is None or v0.device != self.device:
                v0 = torch.ones((Nb,), dtype=torch.float32, device=self.device)
                self._v_cache[Nb] = v0

            # Cost matrix (K x N_b): dist * inv_sigma (faster than div)
            C = self._dist_mat[:, idx] * self._inv_sigma[:, None]

            # Log-domain Sinkhorn for numerical stability
            # logK = -C/eps, stabilized by subtracting column-wise max
            logK = -C / eps
            logK = logK - logK.max(dim=0, keepdim=True).values

            a = pi * float(Nb)  # (K,)
            log_a = torch.log(torch.clamp(a, min=1e-12))
            log_b = torch.zeros((Nb,), dtype=torch.float32, device=self.device)  # log(1)

            log_u = torch.zeros((self.K,), dtype=torch.float32, device=self.device)
            log_v = torch.zeros((Nb,), dtype=torch.float32, device=self.device)

            for _ in range(self.ot_max_iter):
                log_u_prev = log_u

                # log_u = log(a) - logsumexp(logK + log_v, over n)
                log_u = log_a - torch.logsumexp(logK + log_v[None, :], dim=1)

                # log_v = log(b) - logsumexp(logK^T + log_u, over k)
                log_v = log_b - torch.logsumexp(logK.T + log_u[None, :], dim=1)

                # Convergence on log_u (relative)
                denom = torch.clamp(torch.abs(log_u_prev), min=1e-12)
                if torch.max(torch.abs(log_u - log_u_prev) / denom).item() < self.ot_tol:
                    break

            logR = log_u[:, None] + logK + log_v[None, :]
            Rb = torch.exp(logR)

            # Enforce simplex per cell defensively
            Rb = Rb / torch.clamp(Rb.sum(dim=0, keepdim=True), min=1e-12)

            # Incremental delta_R before overwriting
            old_block = self._R[:, idx]
            delta_R += (torch.mean(torch.abs(Rb - old_block)).item() * (float(Nb) / float(self.N)))

            self._R[:, idx] = Rb

        # Update batch diversity statistics (used for diagnostics / lambda estimation)
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

        return delta_R

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False

            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans

        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False

            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects.

        Uses batched sufficient statistics and batched Cholesky solves across clusters,
        and applies the correction via vectorized einsums (no per-cluster Python loops).
        """
        # Precompute shared quantities
        Phi = self._Phi_moe  # (B+1) x N  == (B1 x N)
        Z = self._Z_orig     # d x N
        R = self._R          # K x N

        B1 = self.B + 1

        # S_phi_phi[k] = Phi @ diag(R[k]) @ Phi.T  -> (K, B1, B1)
        S_phi_phi = torch.einsum('bn,kn,cn->kbc', Phi, R, Phi)

        # S_phi_z[k] = Phi @ diag(R[k]) @ Z.T -> (K, B1, d)
        S_phi_z = torch.einsum('bn,kn,dn->kbd', Phi, R, Z)

        # Build ridge penalties (K, B1)
        if self.lambda_estimation:
            # Vectorized: matches find_lambda_torch() behavior exactly
            lamb_mat = torch.zeros((self.K, B1), dtype=torch.float32, device=self.device)
            lamb_mat[:, 1:] = self.alpha * self._E
        else:
            lamb_mat = self._lamb.unsqueeze(0).expand(self.K, -1)

        # A[k] = S_phi_phi[k] + diag(lamb_mat[k])
        # Avoid an extra clone/copy: S_phi_phi is not reused unregularized below.
        A = S_phi_phi
        A.diagonal(dim1=1, dim2=2).add_(lamb_mat)

        # Batched Cholesky solve: A W = S_phi_z  -> W_all: (K, B1, d)
        L = torch.linalg.cholesky(A)
        W_all = torch.cholesky_solve(S_phi_z, L)

        # Do not remove intercept
        W_all[:, 0, :] = 0

        # Tie-to-mean regularizer (shrink cluster-specific W_k toward global mean)
        if self.gamma_w is not None and self.gamma_w > 0:
            W_mean = W_all.mean(dim=0, keepdim=True)  # 1 x B1 x d
            W_all = (W_all + self.gamma_w * W_mean) / (1.0 + self.gamma_w)
            W_all[:, 0, :] = 0  # preserve intercept rule after shrinkage

        # Vectorized correction without materializing (K, d, N):
        # correction[d,n] = sum_k R[k,n] * sum_b W_all[k,b,d] * Phi[b,n]
        correction = torch.einsum('kbd,bn,kn->dn', W_all, Phi, R)  # (d, N)

        self._Z_corr = self._Z_orig - correction

        # Update Z_cos with clamped norms for stability
        _znorm = torch.linalg.norm(self._Z_corr, ord=2, dim=0).clamp_min(1e-12)
        self._Z_cos = self._Z_corr / _znorm


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
