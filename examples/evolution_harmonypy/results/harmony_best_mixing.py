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

        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

        # Cached diversity penalty terms (updated in update_R, reused in compute_objective)
        self._theta_log_cache = None  # (K x B)

        # Precompute one-hot batch assignments for fast gather/scatter ops
        # _batch_id: (N,) int64, each cell's batch index in [0, B-1]
        self._batch_id = torch.argmax(self._Phi, dim=0).to(torch.int64)

        # ---------------------------------------------------------------------
        # Sinkhorn OT warm-start state (per batch):
        # store log-domain dual potentials for stability and speed.
        # Initialized after batch slices are constructed.
        # ---------------------------------------------------------------------
        self._sinkhorn_log_u = None  # list[Tensor(K,)] per batch
        self._sinkhorn_log_v = None  # list[Tensor(nb,)] per batch

        # ---------------------------------------------------------------------
        # Group cells by batch once (permute) to enable contiguous slicing.
        # Keep internal tensors in permuted order for speed; unpermute on output.
        # ---------------------------------------------------------------------
        self._perm = torch.argsort(self._batch_id)
        self._invperm = torch.argsort(self._perm)

        # Permanently reorder internal tensors by batch-grouped permutation
        self._Z_corr = self._Z_corr.index_select(dim=1, index=self._perm)
        self._Z_orig = self._Z_orig.index_select(dim=1, index=self._perm)
        self._Z_cos = self._Z_cos.index_select(dim=1, index=self._perm)
        self._Phi = self._Phi.index_select(dim=1, index=self._perm)
        self._batch_id = self._batch_id.index_select(dim=0, index=self._perm)

        # Re-create Phi_moe with intercept in permuted order (cheaper than permuting)
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

        # Replace per-batch indices with contiguous slices (start, end)
        counts = torch.bincount(self._batch_id, minlength=self.B)
        offsets = torch.zeros(self.B + 1, dtype=torch.int64, device=self.device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        self._batch_slices = [(int(offsets[b].item()), int(offsets[b + 1].item())) for b in range(self.B)]

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
        # Implements batch-balanced assignment as a constraint via per-batch Sinkhorn.
        # ---------------------------------------------------------------------
        self.ot_eps = float(torch.median(self._sigma).item())  # base entropic regularization
        self.ot_eps_min = self.ot_eps / 4.0                    # anneal floor
        self.ot_eps_max = self.ot_eps * 4.0                    # anneal ceiling
        self.ot_max_iter = 25
        self.ot_tol = 1e-3

        # Optional weak diversity bias (not required for mixing because OT enforces marginals)
        self.ot_theta_strength = 0.0

        # Trust-region constraint on correction magnitude (bio conservation)
        # Enforce ||correction_i|| <= rho * ||Z_i|| per cell, with optional annealing.
        self.corr_trust_rho = 0.5
        self.corr_trust_rho_min = 0.25
        self.corr_trust_rho_max = 0.75

        # Tie-to-mean regularizer strength for MOE coefficients (stability / bio conservation)
        self.gamma_w = 1e-2

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()

        # Initialize Sinkhorn warm-start duals per batch slice (in permuted order)
        self._sinkhorn_log_u = [torch.zeros(self.K, dtype=torch.float32, device=self.device) for _ in range(self.B)]
        self._sinkhorn_log_v = []
        for b, (s, e) in enumerate(self._batch_slices):
            nb = max(e - s, 0)
            self._sinkhorn_log_v.append(torch.zeros(nb, dtype=torch.float32, device=self.device))

        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================

    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        Z = self._Z_corr.index_select(dim=1, index=self._invperm)
        return Z.cpu().numpy().T

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        Z = self._Z_orig.index_select(dim=1, index=self._invperm)
        return Z.cpu().numpy().T

    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        Z = self._Z_cos.index_select(dim=1, index=self._invperm)
        return Z.cpu().numpy().T

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        R = self._R.index_select(dim=1, index=self._invperm)
        return R.cpu().numpy().T

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
        Phi = self._Phi.index_select(dim=1, index=self._invperm)
        return Phi.cpu().numpy().T

    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        Phi_moe = self._Phi_moe.index_select(dim=1, index=self._invperm)
        return Phi_moe.cpu().numpy().T

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

        # Preallocate buffers for ridge MOE (avoid per-iteration allocations/clones)
        B1 = self.B + 1
        self._A = torch.zeros((self.K, B1, B1), dtype=torch.float32, device=self.device)

        # Preallocate and reuse large buffers in moe_correct_ridge()
        self._S_phi_z_buf = torch.empty((self.K, B1, self.d), dtype=torch.float32, device=self.device)
        self._correction_buf = torch.empty((self.d, self.N), dtype=torch.float32, device=self.device)

        # Common index tensors (avoid small per-iteration allocations)
        self._diag_idx_B = torch.arange(self.B, device=self.device)
        self._diag_idx_B1 = torch.arange(B1, device=self.device)

        # Preallocate lambda matrix buffer for lambda-estimation mode
        self._lamb_mat_buf = torch.empty((self.K, B1), dtype=torch.float32, device=self.device)

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

        # Cross entropy (R package formula) with numerical stability
        # Compute without materializing a dense (K x N) penalty matrix.
        R_sigma = self._R * self._sigma[:, None]
        theta_log = self._theta_log_cache
        if theta_log is None:
            theta_log = self._compute_diversity_penalty(cache=True)

        _cross_entropy_t = torch.zeros((), dtype=torch.float32, device=self.device)
        for b, (s, e) in enumerate(self._batch_slices):
            if e <= s:
                continue
            _cross_entropy_t = _cross_entropy_t + (R_sigma[:, s:e] * theta_log[:, b:b+1]).sum()
        _cross_entropy = _cross_entropy_t.item()

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

            # Track harmony iteration for annealing schedules (OT epsilon, trust-region rho)
            self._harmony_i = i
            self._harmony_iter_h = iter_harmony

            # No gradients needed anywhere in Harmony
            with torch.no_grad():
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

        rounds = 0
        for i in range(self.max_iter_kmeans):
            # Update Y
            self._Y = self._Z_cos @ self._R.T
            self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)

            # Update distance matrix
            self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

            # Update R
            self._kmeans_i = i
            self.update_R()

            # Compute objective and check convergence (use same rolling window heuristic)
            self.compute_objective()

            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _compute_diversity_penalty(self, cache: bool = True):
        """Compute Harmony diversity penalty terms.

        Returns
        -------
        theta_log : torch.Tensor
            (K x B) tensor with theta[b] * log1p(O/E) per cluster/batch.
        """
        # Clamp to avoid division by zero; use log1p(O/E) for numerical stability
        O_clamped = torch.clamp(self._O, min=1e-8)
        E_clamped = torch.clamp(self._E, min=1e-8)
        OE = O_clamped / E_clamped
        OE = OE.clamp_max(1e3)
        theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log1p(OE)

        if cache:
            self._theta_log_cache = theta_log
        return theta_log

    def update_R(self):
        """Update soft assignments R via per-batch entropic OT (Sinkhorn).

        For each batch b, solve an entropic OT problem between:
          - cluster "supply" a_b = E[:, b] (desired cluster mass within batch)
          - cell "demand"  u_b = 1 per cell in the batch

        Using costs:
          C[k,i] = dist_mat[k,i] / sigma[k]

        This enforces batch-balanced cluster composition as a *constraint* by matching
        row/column marginals of the transport plan.

        Notes
        -----
        - Uses log-domain Sinkhorn with warm-started duals per batch for stability/speed.
        - Optionally adds a weak diversity bias via theta_log if ot_theta_strength > 0,
          but OT already guarantees feasible marginals.
        """
        # Anneal epsilon across Harmony iterations: higher early (smoother), lower later (sharper)
        h_i = getattr(self, "_harmony_i", 1)
        h_T = max(getattr(self, "_harmony_iter_h", self.max_iter_harmony), 1)
        t = float(h_i - 1) / float(max(h_T - 1, 1))
        eps = float(self.ot_eps_max + (self.ot_eps_min - self.ot_eps_max) * t)

        # Optional weak bias (computed once and cached for objective)
        theta_log = self._compute_diversity_penalty(cache=True)
        theta_strength = float(getattr(self, "ot_theta_strength", 0.0))

        # Update R per batch slice with Sinkhorn; keep columns summing to 1 and rows summing to a_b
        for b, (s, e) in enumerate(self._batch_slices):
            nb = e - s
            if nb <= 0:
                continue

            # Costs for this batch: (K x nb)
            C = self._dist_mat[:, s:e] / self._sigma[:, None]

            # If requested, add a weak diversity bias (acts like a per-batch offset cost)
            if theta_strength > 0:
                C = C + theta_strength * theta_log[:, b].unsqueeze(1)

            # Desired cluster masses for this batch (K,), must sum to nb to be feasible.
            a = self._E[:, b].clamp_min(0.0)
            a_sum = a.sum()
            if float(a_sum.item()) <= 1e-8:
                # Degenerate expected mass: fall back to uniform mass over clusters
                a = torch.full((self.K,), float(nb) / float(self.K), dtype=torch.float32, device=self.device)
            else:
                a = a * (float(nb) / a_sum)

            # Uniform per-cell demand, sum to nb
            u = torch.ones(nb, dtype=torch.float32, device=self.device)

            # Log-domain marginals
            log_a = torch.log(a.clamp_min(1e-12))
            log_u = torch.log(u)  # zeros

            # Warm-start duals
            log_u_dual = self._sinkhorn_log_u[b]
            log_v_dual = self._sinkhorn_log_v[b]
            if log_v_dual.numel() != nb:
                log_v_dual = torch.zeros(nb, dtype=torch.float32, device=self.device)
                self._sinkhorn_log_v[b] = log_v_dual

            # logK = -C/eps
            logK = (-C / eps)

            # Sinkhorn iterations in log space:
            #   log_u <- log_a - logsumexp(logK + log_v, dim=1)
            #   log_v <- log_u - logsumexp(logK^T + log_u, dim=0)
            for _ in range(self.ot_max_iter):
                log_u_new = log_a - torch.logsumexp(logK + log_v_dual.unsqueeze(0), dim=1)
                log_v_new = log_u - torch.logsumexp(logK + log_u_new.unsqueeze(1), dim=0)

                du = (log_u_new - log_u_dual).abs().max()
                dv = (log_v_new - log_v_dual).abs().max()

                log_u_dual.copy_(log_u_new)
                log_v_dual.copy_(log_v_new)

                if float(torch.maximum(du, dv).item()) < self.ot_tol:
                    break

            # Transport plan: P = diag(exp(log_u)) * K * diag(exp(log_v))
            # In log form: logP = log_u[:,None] + logK + log_v[None,:]
            P = torch.exp(log_u_dual.unsqueeze(1) + logK + log_v_dual.unsqueeze(0))

            # Numerical cleanup: ensure each cell sums to 1 (mass per cell), without breaking OT too much.
            colsum = P.sum(dim=0, keepdim=True).clamp_min(1e-12)
            P = P / colsum

            # Assign into R
            self._R[:, s:e].copy_(P)

        # Update batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)

        # Vectorized O computation with scatter_add_ (avoid Python loop over batches)
        idx = self._batch_id.unsqueeze(0).expand(self.K, self.N)
        self._O.zero_()
        self._O.scatter_add_(dim=1, index=idx, src=self._R)

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
            return abs(obj_old - obj_new) / max(abs(obj_old), 1e-12) < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects.

        Optimized to exploit one-hot Phi structure:
          - Avoids materializing (K, B1, N)
          - Builds S_phi_phi from cluster mass and O in O(K*B)
          - Computes S_phi_z using 1 GEMM for intercept + per-batch GEMMs
          - Applies correction per batch without (B1, d, N) intermediates
        """
        Z = self._Z_orig  # d x N
        R = self._R       # K x N

        B1 = self.B + 1

        # ------------------------------------------------------------------
        # Compute sufficient statistics S_phi_z (K, B1, d)
        #   row 0 (intercept): sum_n R[k,n] * Z[:,n]
        #   row b+1: sum_{n in batch b} R[k,n] * Z[:,n]
        # ------------------------------------------------------------------
        S_phi_z = self._S_phi_z_buf
        S_phi_z.zero_()

        # Intercept row: (Z @ R.T).T => (K, d)
        S_phi_z[:, 0, :].copy_((Z @ R.T).T)

        # Per-batch rows using contiguous slices
        for b, (s, e) in enumerate(self._batch_slices):
            if e <= s:
                continue
            Zb = Z[:, s:e]
            Rb = R[:, s:e]
            S_phi_z[:, b + 1, :].copy_((Zb @ Rb.T).T)  # (K, d)

        # ------------------------------------------------------------------
        # Structure-aware solve for MOE ridge system (avoid batched Cholesky).
        #
        # For each cluster k:
        #   A00 = s_k + lamb0 (lamb0=0)
        #   A0b = Ab0 = o_b
        #   Abb = o_b + lamb_b  (diagonal), off-diagonal among batches is 0
        #
        # Solve via Schur complement with only elementwise ops / reductions.
        # ------------------------------------------------------------------
        s_k = R.sum(dim=1)          # (K,)
        O = self._O                 # (K, B)

        # Build ridge penalties (K, B1)
        if self.lambda_estimation:
            lamb_mat = self._lamb_mat_buf
            lamb_mat.zero_()
            lamb_mat[:, 1:].copy_(self._E * self.alpha)
            lamb_mat[:, 0] = 0
        else:
            lamb_mat = self._lamb.unsqueeze(0).expand(self.K, -1)

        # Diagonal inverse for batch block: D = diag(O + lamb_batch)
        D_inv = 1.0 / (O + lamb_mat[:, 1:]).clamp_min(1e-12)  # (K, B)

        # Schur complement: schur = A00 - sum_b o_b^2 / D_bb
        A00 = s_k + lamb_mat[:, 0]  # lamb0 is 0
        schur = A00 - (O * O * D_inv).sum(dim=1)
        schur = schur.clamp_min(1e-8)  # stability for tiny clusters

        # RHS:
        # rhs0 = b0 - sum_b o_b / D_bb * bb
        b0 = S_phi_z[:, 0, :]       # (K, d)
        bb = S_phi_z[:, 1:, :]      # (K, B, d)

        rhs0 = b0 - (O * D_inv)[:, :, None].mul(bb).sum(dim=1)  # (K, d)

        # w0 (K, d)
        w0 = rhs0 / schur[:, None]

        # wb (K, B, d)
        wb = D_inv[:, :, None] * (bb - O[:, :, None] * w0[:, None, :])  # (K, B, d)

        # Pack into W_all for downstream logic / shrinkage; intercept will be zeroed anyway.
        W_all = torch.empty((self.K, B1, self.d), dtype=torch.float32, device=self.device)
        W_all[:, 0, :].copy_(w0)
        W_all[:, 1:, :].copy_(wb)

        # Do not remove intercept
        W_all[:, 0, :] = 0

        # Tie-to-mean regularizer (shrink cluster-specific W_k toward global mean)
        # Use cluster-size adaptive shrinkage: stronger for small clusters, weaker for big ones.
        if self.gamma_w is not None and self.gamma_w > 0:
            W_mean = W_all.mean(dim=0, keepdim=True)  # 1 x B1 x d

            eps = 1e-8
            gamma0 = float(self.gamma_w)
            gamma_k = gamma0 * (s_k.mean() / s_k.clamp_min(eps))
            gamma_k = gamma_k.clamp(min=0.0, max=100.0)

            W_all = (W_all + gamma_k[:, None, None] * W_mean) / (1.0 + gamma_k[:, None, None])
            W_all[:, 0, :] = 0  # preserve intercept rule after shrinkage

        # ------------------------------------------------------------------
        # Apply correction without (B1,d,N) or einsums, using one-hot batch structure.
        # For each batch b: correction[:, idx] = (Wb.T @ R[:, idx])
        # ------------------------------------------------------------------
        correction = self._correction_buf
        correction.zero_()

        for b, (s, e) in enumerate(self._batch_slices):
            if e <= s:
                continue
            Wb = W_all[:, b + 1, :]  # (K, d)
            correction[:, s:e] = Wb.T @ R[:, s:e]  # (d, nb)

        # ------------------------------------------------------------------
        # Trust-region / proximal constraint on correction magnitude (per cell)
        # Enforce ||corr_i||_2 <= rho * ||Z_i||_2 by rescaling when violated.
        # Optionally anneal rho across Harmony iterations.
        # ------------------------------------------------------------------
        h_i = getattr(self, "_harmony_i", 1)
        h_T = max(getattr(self, "_harmony_iter_h", self.max_iter_harmony), 1)
        t = float(h_i - 1) / float(max(h_T - 1, 1))
        rho = float(self.corr_trust_rho_min + (self.corr_trust_rho_max - self.corr_trust_rho_min) * t)
        rho = max(0.0, rho)

        if rho > 0:
            corr_norm = torch.linalg.norm(correction, ord=2, dim=0).clamp_min(1e-12)
            z_norm = torch.linalg.norm(self._Z_orig, ord=2, dim=0).clamp_min(1e-12)
            max_corr = rho * z_norm
            scale = torch.minimum(torch.ones_like(corr_norm), max_corr / corr_norm)
            correction.mul_(scale.unsqueeze(0))

        self._Z_corr = self._Z_orig - correction

        # Update Z_cos with clamped norms for stability
        _znorm = torch.linalg.norm(self._Z_corr, ord=2, dim=0).clamp_min(1e-12)
        self._Z_cos = self._Z_corr / _znorm


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x <= 0 (avoids log(0))."""
    return torch.where(x > 0, x * torch.log(x), torch.zeros_like(x))


# NOTE: harmony_pow_torch was removed as unused dead code (and was a slow Python loop).


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
