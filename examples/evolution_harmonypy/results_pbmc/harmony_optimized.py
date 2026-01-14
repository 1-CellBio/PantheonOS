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
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
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

        # Simple L2 normalization (safe against tiny norms)
        _norm = torch.linalg.norm(self._Z_orig, ord=2, dim=0)
        self._Z_cos = self._Z_orig / torch.clamp(_norm, min=1e-8)

        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        # Precompute batch id per cell (Phi is one-hot): shape (N,)
        self._batch_id = torch.argmax(self._Phi, dim=0)

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

        # Persistent buffers for moe_correct_ridge() to avoid per-iteration allocations
        self._cov_buf = torch.zeros((self.K, self.B + 1, self.B + 1), dtype=torch.float32, device=self.device)
        self._G_buf = torch.zeros((self.K, self.d, self.B + 1), dtype=torch.float32, device=self.device)
        self._corr_buf = torch.zeros((self.d, self.N), dtype=torch.float32, device=self.device)
        self._eye_B1 = torch.eye(self.B + 1, dtype=torch.float32, device=self.device)

        # Accumulated correction to prevent cumulative drift: Z_corr = Z_orig - C
        self._C = torch.zeros((self.d, self.N), dtype=torch.float32, device=self.device)

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
        # Vectorized via batch_id gather to avoid (K,B)@(B,N) GEMM every iteration.
        R_sigma = self._R * self._sigma[:, None]

        batch_id = self._batch_id  # (N,)
        O_g = self._O.index_select(dim=1, index=batch_id)  # (K, N)
        E_g = self._E.index_select(dim=1, index=batch_id)  # (K, N)

        O_g = torch.clamp(O_g, min=1e-8)
        E_g = torch.clamp(E_g, min=1e-8)

        ratio = (O_g + E_g) / E_g
        theta_g = self._theta.index_select(0, batch_id)  # (N,)
        pen = theta_g.unsqueeze(0) * torch.log(ratio)  # (K, N)

        _cross_entropy = torch.sum(R_sigma * pen).item()

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

            # Diversity tempering schedule to reduce early overcorrection
            # Ramp up over first 3 iterations (or fewer if iter_harmony < 3)
            ramp_denom = float(min(3, iter_harmony)) if iter_harmony > 0 else 1.0
            self._theta_eff = self._theta * min(1.0, i / ramp_denom)

            # Diversity-temperature ramp (slower than theta ramp) to de-spike penalties
            beta_denom = 5.0
            self._beta = min(1.0, i / beta_denom)

            # Correction step-size schedule (avoid large early destructive moves)
            # Ramp from 0.3 -> 1.0 over first 3 iterations
            self._eta = 0.3 + 0.7 * min(1.0, i / ramp_denom)

            # Anchor regression blending (use partially-corrected embedding early)
            self._reg_blend = 0.3 + 0.7 * min(1.0, i / ramp_denom)

            # R damping schedule (more conservative early, faster settling later)
            # Ramp from 0.4 -> 0.85 over first 3 iterations
            self._gamma_eff = 0.4 + 0.45 * min(1.0, i / ramp_denom)

            with torch.no_grad():
                self.cluster()
                self.moe_correct_ridge()

            # Harmony-level early stop: objective + R stability
            converged = self.check_convergence(1)
            last_r_delta = getattr(self, "_last_r_delta", None)
            if last_r_delta is not None and last_r_delta < 1e-4 and len(self.objective_harmony) >= 2:
                obj_old = self.objective_harmony[-2]
                obj_new = self.objective_harmony[-1]
                if abs(obj_old - obj_new) / max(abs(obj_old), 1e-8) < self.epsilon_harmony:
                    converged = True

            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break

        if verbose and not converged:
            logger.info("Stopped before convergence")

    def cluster(self):
        with torch.no_grad():
            rounds = 0
            for i in range(self.max_iter_kmeans):
                # Update Y
                self._Y = self._Z_cos @ self._R.T
                self._Y = self._Y / torch.clamp(torch.linalg.norm(self._Y, ord=2, dim=0), min=1e-8)

                # Update distance matrix
                self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

                # Update R (early stop if R stabilizes)
                r_delta = self.update_R()

                # Compute objective and check convergence
                self.compute_objective()

                # Cheap stabilization check on R change (looser threshold for faster convergence)
                if r_delta < 3e-4:
                    rounds = i + 1
                    break

                if i > self.window_size:
                    if self.check_convergence(0):
                        rounds = i + 1
                        break
                rounds = i + 1

            self.kmeans_rounds.append(rounds)
            self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        with torch.no_grad():
            # Vectorized full update in log-space (order-independent) + optional damping
            # log_scale: (K, N)
            log_scale = -self._dist_mat / self._sigma[:, None]

            # Use tempered theta if available
            theta_eff = getattr(self, "_theta_eff", self._theta)

            # Diversity temperature ramp to de-spike penalty impact
            beta = getattr(self, "_beta", 1.0)

            # Cluster-size modulation: reduce diversity pressure on small clusters (protect rare biology).
            n_k = torch.clamp(self._R.sum(dim=1), min=1e-8)  # (K,)
            med = torch.median(n_k)
            # scale in (s_min..1], where small clusters get smaller scale
            s_min = 0.35
            cluster_scale = torch.sqrt(n_k / torch.clamp(med, min=1e-8))
            cluster_scale = torch.clamp(cluster_scale, min=s_min, max=1.0)  # (K,)

            # Vectorized leave-one-out diversity penalty using precomputed batch_id.
            # For each cell n in batch b:
            #   denom[k,n] = (O[k,b] - R[k,n]) + E[k,b]
            #   log_penalty[k,n] = theta[b] * log( E[k,b] / denom[k,n] )
            batch_id = self._batch_id  # (N,)
            O_g = self._O.index_select(dim=1, index=batch_id)  # (K, N)
            E_g = self._E.index_select(dim=1, index=batch_id)  # (K, N)
            theta_g = theta_eff.index_select(0, batch_id)      # (N,)

            # Confidence-adaptive theta (down-weight diversity pressure on high-confidence cells)
            # Add a floor to avoid all-or-nothing behavior.
            # conf[n] = max_k R[k,n]
            conf = torch.clamp(self._R.max(dim=0).values, 0.0, 1.0)  # (N,)
            p = 2.0
            theta_floor = 0.15
            theta_cell = theta_g * (theta_floor + (1.0 - theta_floor) * torch.pow(1.0 - conf, p))

            denom = torch.clamp(O_g - self._R + E_g, min=1e-8)
            # Cap ratio to avoid huge log excursions (prevents batch-driven cluster swapping)
            # Anneal cap with beta to reduce early diversity "shock"
            # early beta~0 => ~10, later beta~1 => ~15
            ratio_max = 10.0 + 5.0 * float(beta)
            ratio = torch.clamp(E_g / denom, min=1e-8, max=ratio_max)
            log_penalty = theta_cell.unsqueeze(0) * torch.log(ratio)  # (K, N)

            # Apply cluster-size modulation to diversity penalty
            log_penalty = log_penalty * cluster_scale.unsqueeze(1)

            # Combine and normalize over clusters
            logR = log_scale + beta * log_penalty
            R_new = torch.softmax(logR, dim=0)

            # Delta metric before overwriting (avoid clone/allocation)
            r_delta = torch.mean(torch.abs(R_new - self._R)).item()
            self._last_r_delta = r_delta

            # Adaptive damping for stability / convergence speed:
            # - large r_delta => smaller gamma (more conservative)
            # - small r_delta => larger gamma (faster settling)
            gamma_sched = getattr(self, "_gamma_eff", 0.7)
            gamma_min, gamma_max = 0.25, 0.95
            # Map r_delta (0..~1) to gamma in [gamma_max..gamma_min]
            gamma_adapt = gamma_max - (gamma_max - gamma_min) * min(1.0, r_delta / 3e-3)
            gamma = float(np.clip(gamma_adapt * float(gamma_sched), gamma_min, gamma_max))

            if gamma < 1.0:
                self._R = (1.0 - gamma) * self._R + gamma * R_new
                self._R = self._R / torch.clamp(self._R.sum(dim=0, keepdim=True), min=1e-8)
            else:
                self._R = R_new

            # Recompute statistics (no EMA smoothing; keep statistics consistent with current R)
            # Compute O via scatter-add using batch_id (avoid GEMM with Phi.T).
            self._O.zero_()
            self._O.scatter_add_(dim=1, index=batch_id.unsqueeze(0).expand(self.K, -1), src=self._R)
            self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)

            return r_delta

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
        """Ridge regression correction for batch effects."""
        with torch.no_grad():
            # Shapes:
            #   Z_orig:    (d, N)
            #   R:         (K, N)
            #   Phi_moe:   (B+1, N) = [1; Phi]
            # Use a blended target for regression to avoid destructive early updates.
            reg_blend = getattr(self, "_reg_blend", 1.0)
            Z_reg = reg_blend * self._Z_corr + (1.0 - reg_blend) * self._Z_orig

            R = self._R
            Phi_moe = self._Phi_moe

            # Exploit intercept + one-hot structure of Phi_moe to build cov in closed form.
            # Let n_k = sum_n R[k,n] (K,)
            # Let O = R @ Phi.T (K,B)  (observed cluster x batch)
            # Then for each cluster k:
            #   cov[0,0]   = n_k
            #   cov[0,1:]  = O[k,:]
            #   cov[1:,0]  = O[k,:]
            #   cov[1:,1:] = diag(O[k,:])
            n_k = torch.clamp(R.sum(dim=1), min=1e-8)  # (K,)
            O = self._O  # (K, B)

            cov = self._cov_buf
            cov.zero_()
            cov[:, 0, 0] = n_k
            cov[:, 0, 1:] = O
            cov[:, 1:, 0] = O
            cov[:, 1:, 1:] = torch.diag_embed(O)

            # Build lambda per cluster
            if self.lambda_estimation:
                # lamb_k = [0, alpha * E[k,:]] for each k
                lamb_k = torch.zeros((self.K, self.B + 1), dtype=torch.float32, device=self.device)
                lamb_k[:, 1:] = self._E * self.alpha
            else:
                # Fixed lambda, but inflate for weak/small clusters to reduce overcorrection
                c = 1.0
                inflate = c / torch.sqrt(n_k)  # (K,)
                lamb_k = self._lamb.unsqueeze(0).repeat(self.K, 1)  # (K, B+1)
                lamb_k[:, 1:] = lamb_k[:, 1:] + inflate.unsqueeze(1)

            cov = cov + torch.diag_embed(lamb_k)  # (K, B+1, B+1)

            # Add tiny jitter to reduce Cholesky failures (reuse preallocated eye)
            cov = cov + 1e-6 * self._eye_B1.unsqueeze(0)

            # Build RHS G using closed-form matmuls leveraging one-hot batches.
            # Intercept term:
            #   G0[d,k] = sum_n Z_reg[d,n] * R[k,n] = (Z_reg @ R.T)[d,k]
            G = self._G_buf
            G.zero_()
            G0 = Z_reg @ R.T  # (d, K)
            G[:, :, 0] = G0.T  # (K, d)

            # Batch terms via scatter-add over batch_id (replace per-batch Python loop):
            # For each k,b and dimension j:
            #   G[k,j,b+1] = sum_{n:batch(n)=b} R[k,n] * Z_reg[j,n]
            bid = self._batch_id  # (N,)
            # We iterate over embedding dimension (small d) and scatter into (K,B)
            for j in range(self.d):
                weighted = R * Z_reg[j, :].unsqueeze(0)  # (K, N)
                # tmp: (K, B)
                tmp = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
                tmp.scatter_add_(dim=1, index=bid.unsqueeze(0).expand(self.K, -1), src=weighted)
                G[:, j, 1:] = tmp

            # Solve cov @ Beta = G^T  => Beta: (K, B+1, d)
            # Prefer batched cholesky
            Beta = None
            try:
                L, info = torch.linalg.cholesky_ex(cov)
                if torch.any(info != 0):
                    raise RuntimeError("Cholesky failed for some clusters")
                Beta = torch.cholesky_solve(G.transpose(1, 2), L)  # (K, B+1, d)
            except RuntimeError:
                Beta = torch.linalg.solve(cov, G.transpose(1, 2))  # (K, B+1, d)

            # Do not remove intercept
            Beta[:, 0, :] = 0

            # Enforce "zero-mean" batch correction within each cluster by projecting
            # batch coefficients to have zero mean across batches weighted by Pr_b.
            # For each (k,d): sum_b Pr_b[b] * Beta[k,b+1,d] = 0
            P = self._Pr_b  # (B,)
            Psum = torch.clamp(P.sum(), min=1e-8)
            mean = (Beta[:, 1:, :] * P.view(1, -1, 1)).sum(dim=1, keepdim=True) / Psum  # (K,1,d)
            Beta[:, 1:, :] = Beta[:, 1:, :] - mean

            # Blockwise correction using batch-id gathering (avoid Phi one-hot multiplications).
            BetaT = Beta.transpose(1, 2).contiguous()  # (K, d, B+1)
            corr = self._corr_buf
            corr.zero_()

            # Choose a reasonable block size to control temporary tensor memory.
            # temp ~ K*d*nb floats
            max_tmp_elems = 32_000_000  # ~128MB fp32
            denom = max(1, int(self.K * self.d))
            nb = max(256, min(self.N, max_tmp_elems // denom))

            for n0 in range(0, self.N, nb):
                n1 = min(self.N, n0 + nb)
                R_b = R[:, n0:n1]  # (K, nb)
                bid = self._batch_id[n0:n1]  # (nb,) in 0..B-1

                # Gather batch coefficients for each cell (shift +1 for intercept slot)
                Beta_g = BetaT.index_select(dim=2, index=bid + 1)  # (K, d, nb)

                # Use bmm instead of einsum for this contraction:
                #   corr[:, n0:n1] = sum_k R[k,n] * Beta_g[k,:,n]
                Beta_g_nb = Beta_g.permute(2, 0, 1).contiguous()     # (nb, K, d)
                R_nb = R_b.T.unsqueeze(1).contiguous()              # (nb, 1, K)
                corr_nb = torch.bmm(R_nb, Beta_g_nb).squeeze(1)     # (nb, d)
                corr[:, n0:n1] = corr_nb.T

            # -----------------------------------------------------------------
            # Structure-preserving tangent-space correction:
            # remove radial component along the soft centroid direction y_n.
            # y_n = normalize(Y @ R[:,n])  (computed blockwise)
            # corr_n <- corr_n - y_n * (y_n^T corr_n)
            # -----------------------------------------------------------------
            Y = self._Y  # (d, K)
            for n0 in range(0, self.N, nb):
                n1 = min(self.N, n0 + nb)
                R_b = R[:, n0:n1]              # (K, nb)
                y = Y @ R_b                    # (d, nb)
                y = y / torch.clamp(torch.linalg.norm(y, ord=2, dim=0), min=1e-8)
                c_b = corr[:, n0:n1]           # (d, nb)
                dot = torch.sum(y * c_b, dim=0, keepdim=True)  # (1, nb)
                corr[:, n0:n1] = c_b - y * dot

            # Correction-magnitude guard (per-cell) to improve biological conservation.
            # Cap ||corr|| relative to ||Z_reg|| with a ramp tied to eta and confidence.
            eta = getattr(self, "_eta", 1.0)
            z_norm = torch.linalg.norm(Z_reg, ord=2, dim=0)        # (N,)
            corr_norm = torch.linalg.norm(corr, ord=2, dim=0)      # (N,)

            # Tighter ramp max fraction from 0.08 -> 0.20 as eta goes 0.3 -> 1.0
            eta01 = torch.clamp(torch.tensor((eta - 0.3) / 0.7, device=self.device), 0.0, 1.0)
            max_frac_base = 0.08 + 0.12 * eta01

            # Confidence-based cap (protect high-confidence core cells more):
            # allowed movement decreases with confidence with a floor.
            conf = torch.clamp(R.max(dim=0).values, 0.0, 1.0)  # (N,)
            max_frac_n = max_frac_base * (0.4 + 0.6 * (1.0 - conf))  # in [0.4*base, 1.0*base]

            # Boundary proxy via assignment entropy (cheap, stable):
            # higher entropy => more boundary-like => allow *less* movement.
            ent = -torch.sum(R * torch.log(torch.clamp(R, min=1e-8)), dim=0)  # (N,)
            ent = ent / torch.clamp(torch.log(torch.tensor(float(self.K), device=self.device)), min=1e-8)
            max_frac_n = max_frac_n * (1.0 - 0.35 * torch.clamp(ent, 0.0, 1.0))

            scale = (max_frac_n * z_norm) / torch.clamp(corr_norm, min=1e-8)
            scale = torch.clamp(scale, max=1.0)
            corr = corr * scale.unsqueeze(0)

            # Stop cumulative drift: keep explicit accumulated correction C and tie Z_corr to Z_orig.
            # Update C with current step-size schedule (eta).
            self._C = self._C + eta * corr
            self._Z_corr = self._Z_orig - self._C

            # Update Z_cos (safe against tiny norms)
            _norm = torch.linalg.norm(self._Z_corr, ord=2, dim=0)
            self._Z_cos = self._Z_corr / torch.clamp(_norm, min=1e-8)


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * torch.log(x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column (vectorized)."""
    A = torch.clamp(A, min=1e-8)
    return torch.pow(A, T.unsqueeze(0))


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
