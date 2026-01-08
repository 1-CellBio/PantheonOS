# File: harmony.py
"""
Harmony Algorithm for Data Integration.

This is a simplified implementation of the Harmony algorithm for integrating
multiple high-dimensional datasets. It uses fuzzy k-means clustering and
linear corrections to remove batch effects while preserving biological structure.

Reference:
    Korsunsky et al., "Fast, sensitive and accurate integration of single-cell
    data with Harmony", Nature Methods, 2019.

This implementation is designed to be optimized by Pantheon Evolution.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List

try:
    from scipy.linalg import cho_factor, cho_solve
except Exception:  # scipy is optional; fall back to np.linalg.solve
    cho_factor = None
    cho_solve = None


class Harmony:
    """
    Harmony algorithm for batch effect correction.

    Attributes:
        Z_corr: Corrected embedding after harmonization
        Z_orig: Original embedding
        R: Soft cluster assignments (cells x clusters)
        objectives: History of objective function values
    """

    def __init__(
        self,
        n_clusters: int = 100,
        theta: float = 2.0,
        sigma: float = 0.1,
        lamb: float = 1.0,
        max_iter: int = 10,
        max_iter_kmeans: int = 20,
        epsilon_cluster: float = 1e-5,
        epsilon_harmony: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Harmony.

        Args:
            n_clusters: Number of clusters for k-means
            theta: Diversity clustering penalty parameter
            sigma: Width of soft k-means clusters
            lamb: Ridge regression penalty
            max_iter: Maximum iterations of Harmony algorithm
            max_iter_kmeans: Maximum iterations for clustering step
            epsilon_cluster: Convergence threshold for clustering
            epsilon_harmony: Convergence threshold for Harmony
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.theta = theta
        self.sigma = sigma
        self.lamb = lamb
        self.max_iter = max_iter
        self.max_iter_kmeans = max_iter_kmeans
        self.epsilon_cluster = epsilon_cluster
        self.epsilon_harmony = epsilon_harmony
        self.random_state = random_state

        # Will be set during fit
        self.Z_orig = None
        self.Z_corr = None
        self.R = None
        self.Y = None  # Cluster centroids
        self.Phi = None  # Batch membership matrix
        self.objectives = []

        # Internal dampening to reduce overcorrection (private; does not change public API)
        self._correction_alpha = 0.5
        self._epsilon_z = 1e-4  # embedding-based early stop threshold on mean ||delta||

        # Cached / precomputed internals for performance
        self._design = None  # (n_cells x p) with intercept + batch indicators (excluding ref batch)
        self._design_T = None  # (p x n_cells) cached transpose for fast GEMMs in _correct
        self._I_ridge = None  # (p x p) identity for ridge penalty in _correct
        self._Z_sq = None  # (n_cells,) cached ||Z_corr||^2 per cell
        self._Y_sq = None  # (n_clusters,) cached ||Y||^2 per cluster
        self._last_dist = None  # (n_clusters x n_cells) last computed distances

        # Optional cached transpose of Phi for fast O computation (N x B)
        self._Phi_T = None

        # Cache diversity stats from _update_R to avoid recomputation in _compute_objective
        self._O_last = None  # (n_clusters x n_batches) last cluster-by-batch proportions
        self._log_ratio_last = None  # (n_clusters x n_batches) last log-ratio for diversity penalty

        # Lagged diversity penalty state (used in _update_R single-pass update)
        self._log_ratio_prev = None  # (n_clusters x n_batches) EMA-smoothed log-ratio for diversity penalty

        # Diversity penalty stabilization (private; does not change public API)
        self._diversity_ema_gamma = 0.3
        self._diversity_clip = 5.0

        # Effective theta (annealed over outer Harmony iterations)
        self._theta_eff = theta

        # Batch indexing cache (avoid dense Phi multiplies in hot loops)
        self._batch_id = None  # (n_cells,) int in [0, n_batches)
        self._batch_indices = None  # List[np.ndarray] indices per batch id
        self._n_batches = None

        # Preallocated buffers (hot-loop allocations in _update_R / _compute_distances)
        self._buf_logR = None  # (n_clusters x n_cells)
        self._buf_R = None  # (n_clusters x n_cells)
        self._buf_cross = None  # (n_clusters x n_cells) reusable Y @ Z^T
        self._buf_dist = None  # (n_clusters x n_cells) reusable distances

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> "Harmony":
        """
        Fit Harmony to the data.

        Args:
            X: Data matrix (n_cells x n_features), typically PCA coordinates
            batch_labels: Batch labels for each cell (n_cells,)

        Returns:
            self with Z_corr containing corrected coordinates
        """
        n_cells, n_features = X.shape

        # Store original
        self.Z_orig = X.copy()
        self.Z_corr = X.copy()

        # Create batch membership matrix (one-hot encoding)
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        self._n_batches = n_batches

        # Map batch labels to contiguous ids [0..n_batches-1] and cache indices per batch
        batch_to_id = {b: i for i, b in enumerate(unique_batches)}
        self._batch_id = np.array([batch_to_id[b] for b in batch_labels], dtype=np.int64)
        self._batch_indices = [np.where(self._batch_id == b)[0] for b in range(n_batches)]

        self.Phi = np.zeros((n_batches, n_cells))
        for i in range(n_batches):
            self.Phi[i, self._batch_indices[i]] = 1

        # Compute batch proportions
        self.batch_props = self.Phi.sum(axis=1) / n_cells

        # Precompute design matrix once: intercept + batch indicators (drop first batch as reference)
        self._design = np.column_stack(
            [
                np.ones(n_cells),
                self.Phi[1:, :].T,
            ]
        )  # (n_cells x p), where p = 1 + (n_batches - 1)

        # Cache transpose/identity for fast weighted least squares in _correct()
        self._design_T = self._design.T
        self._I_ridge = np.eye(self._design.shape[1], dtype=np.float64)

        # Cache Phi transpose for fast responsibility-by-batch aggregation
        self._Phi_T = self.Phi.T  # (n_cells x n_batches)

        # Initialize lagged diversity stats
        self._log_ratio_prev = None

        # Initialize cached norms
        self._Z_sq = np.sum(self.Z_corr ** 2, axis=1)  # (n_cells,)

        # Initialize clusters
        self._init_clusters()

        # Main Harmony loop
        self.objectives = []
        prev_mean_delta = None
        for iteration in range(self.max_iter):
            # Anneal diversity penalty to preserve biology early, enforce mixing later
            # Use a slower ramp: theta_eff = theta * t^2, where t in [0, 1]
            t = iteration / max(1, self.max_iter - 1)
            self._theta_eff = float(self.theta) * (t ** 2)

            # Clustering step
            self._cluster()

            # Correction step
            mean_delta = self._correct()

            # Adaptive correction strength: speed up when stabilizing, damp when diverging
            if mean_delta is not None:
                if prev_mean_delta is not None and prev_mean_delta > 0:
                    if mean_delta < prev_mean_delta * 0.9:
                        self._correction_alpha = min(self._correction_alpha * 1.05, 1.0)
                    elif mean_delta > prev_mean_delta * 1.1:
                        self._correction_alpha = max(self._correction_alpha * 0.7, 0.2)
                prev_mean_delta = mean_delta

            # Check convergence
            obj = self._compute_objective()
            self.objectives.append(obj)

            # Embedding-based early stop (often more directly tied to correction stability)
            if mean_delta is not None and mean_delta < self._epsilon_z:
                break

            if iteration > 0:
                obj_change = abs(self.objectives[-2] - self.objectives[-1])
                if obj_change < self.epsilon_harmony:
                    break

        return self

    def _init_clusters(self):
        """Initialize cluster centroids using k-means."""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=1,
            max_iter=25,
        )
        kmeans.fit(self.Z_corr)
        # Store centroids as (n_clusters x n_features) for faster distance GEMMs (KxN layout)
        self.Y = kmeans.cluster_centers_  # (n_clusters x n_features)
        self._Y_sq = np.sum(self.Y ** 2, axis=1)  # (n_clusters,)

        # Initialize soft assignments
        self._update_R()

    def _cluster(self):
        """Run clustering iterations."""
        # Clear cached diversity stats at the start of clustering iterations to avoid stale objective terms
        self._O_last = None
        self._log_ratio_last = None

        # Avoid full copies of R (K x N) and Y (F x K) each iteration by keeping scalar summaries
        prev_Y_ss = float(np.sum(self.Y ** 2)) if self.Y is not None else None
        prev_R_mean = float(self.R.mean()) if self.R is not None else None

        for _ in range(self.max_iter_kmeans):
            # Update centroids
            self._update_centroids()

            # Update soft assignments
            self._update_R()

            # Convergence checks using cheap summaries (no large allocations)
            if prev_R_mean is not None:
                r_mean = float(self.R.mean())
                r_change = abs(r_mean - prev_R_mean)
                prev_R_mean = r_mean
                if r_change < self.epsilon_cluster:
                    break

            if prev_Y_ss is not None:
                y_ss = float(np.sum(self.Y ** 2))
                y_change = abs(y_ss - prev_Y_ss) / (prev_Y_ss + 1e-12)
                prev_Y_ss = y_ss
                if y_change < self.epsilon_cluster:
                    break

    def _update_centroids(self):
        """Update cluster centroids."""
        # Weighted average of cells
        weights = self.R  # (n_clusters x n_cells)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8  # (K x 1)

        # Y = (R @ Z) / sum(R)  -> (K x F)
        self.Y = (weights @ self.Z_corr) / weights_sum
        self._Y_sq = np.sum(self.Y ** 2, axis=1)  # (n_clusters,)

    def _update_R(self):
        """Update soft cluster assignments with diversity penalty (single-pass softmax with lagged/EMA penalty)."""
        # Compute distances to centroids
        # dist[k, i] = ||z_i - y_k||^2
        dist = self._compute_distances()
        self._last_dist = dist

        theta_eff = float(getattr(self, "_theta_eff", self.theta))

        # Preallocate / reuse buffers to reduce allocations in hot loop
        if self._buf_logR is None or self._buf_logR.shape != dist.shape:
            self._buf_logR = np.empty_like(dist, dtype=np.float64)
        if self._buf_R is None or self._buf_R.shape != dist.shape:
            self._buf_R = np.empty_like(dist, dtype=np.float64)

        logR = self._buf_logR
        R = self._buf_R

        # Base log responsibilities (in-place)
        logR[:] = -(dist / self.sigma)

        # Incorporate lagged diversity penalty in log-space (more stable than probability-space scaling)
        if theta_eff > 0 and self._log_ratio_prev is not None:
            logR -= theta_eff * self._log_ratio_prev[:, self._batch_id]

        # Single stabilized softmax (in-place)
        logR -= logR.max(axis=0, keepdims=True)
        np.exp(logR, out=R)
        R /= (R.sum(axis=0, keepdims=True) + 1e-8)
        self.R = R

        # Update diversity stats for objective reuse and for next iteration's lagged penalty
        if theta_eff <= 0:
            self._O_last = None
            self._log_ratio_last = None
            self._log_ratio_prev = None
            return

        R_sum = R.sum(axis=1, keepdims=True) + 1e-8  # (K x 1)
        B = self._n_batches if self._n_batches is not None else self.Phi.shape[0]

        # Vectorized O_num via GEMM when Phi_T cached; fallback to cached indices
        if self._Phi_T is not None:
            O_num = R @ self._Phi_T  # (K x B)
        else:
            O_num = np.zeros((self.n_clusters, B), dtype=R.dtype)
            for b in range(B):
                idx = self._batch_indices[b] if self._batch_indices is not None else np.where(self._batch_id == b)[0]
                if idx.size == 0:
                    continue
                O_num[:, b] = R[:, idx].sum(axis=1)

        O = O_num / R_sum  # (K x B)
        expected = self.batch_props[np.newaxis, :]  # (1 x B)
        log_ratio = np.log((O + 1e-8) / (expected + 1e-8))  # (K x B)

        # Clip extreme penalties to prevent late-iteration blow-ups
        clip = float(getattr(self, "_diversity_clip", 0.0) or 0.0)
        if clip > 0:
            log_ratio = np.clip(log_ratio, -clip, clip)

        self._O_last = O
        self._log_ratio_last = log_ratio

        # Smooth updates with EMA to reduce oscillations
        gamma = float(getattr(self, "_diversity_ema_gamma", 0.3))
        if self._log_ratio_prev is None:
            self._log_ratio_prev = log_ratio
        else:
            self._log_ratio_prev = (1.0 - gamma) * self._log_ratio_prev + gamma * log_ratio

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids using cached norms when possible (returns K x N)."""
        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 * y @ z
        if self._Z_sq is None:
            self._Z_sq = np.sum(self.Z_corr ** 2, axis=1)  # (n_cells,)
        if self._Y_sq is None:
            self._Y_sq = np.sum(self.Y ** 2, axis=1)  # (n_clusters,)

        K = self.Y.shape[0]
        N = self.Z_corr.shape[0]

        # Preallocate / reuse large buffers to reduce allocations
        if self._buf_cross is None or self._buf_cross.shape != (K, N):
            self._buf_cross = np.empty((K, N), dtype=np.float64)
        if self._buf_dist is None or self._buf_dist.shape != (K, N):
            self._buf_dist = np.empty((K, N), dtype=np.float64)

        cross = self._buf_cross
        dist = self._buf_dist

        # cross = Y @ Z.T -> (K x N) into reusable buffer
        # np.matmul supports out= for ndarray operands
        np.matmul(self.Y, self.Z_corr.T, out=cross)

        # dist = ||y||^2 + ||z||^2 - 2*cross (in-place)
        dist[:] = self._Y_sq[:, None]
        dist += self._Z_sq[None, :]
        dist -= 2.0 * cross
        return dist

    def _correct(self) -> Optional[float]:
        """
        Apply correction to remove batch effects using per-cluster weighted ridge regression (WLS),
        exploiting the special design structure (intercept + one-hot batches).

        Uses sufficient statistics per (cluster, batch) to avoid building large Dw/Zw intermediates.

        Returns mean ||applied_delta|| per cell (aligned with the actual applied step size).
        """
        n_cells, n_features = self.Z_corr.shape

        if self._batch_indices is None or self._n_batches is None:
            # Fallback: ensure batch indexing exists (should be set in fit)
            unique_batches = np.unique(self._batch_id) if self._batch_id is not None else np.unique(
                np.argmax(self.Phi, axis=0)
            )
            self._n_batches = len(unique_batches)
            if self._batch_id is None:
                self._batch_id = np.argmax(self.Phi, axis=0).astype(np.int64)
            self._batch_indices = [np.where(self._batch_id == b)[0] for b in range(self._n_batches)]

        B = int(self._n_batches)
        p = 1 + max(0, B - 1)  # intercept + batch indicators (excluding reference batch 0)

        R = self.R  # (K x N)
        Z = self.Z_corr  # (N x F)
        K = int(self.n_clusters)

        # Scale-aware small-cluster gating to reduce unstable corrections
        min_cluster_mass = max(5.0, 0.01 * (n_cells / max(1, K)))

        # Sufficient statistics:
        #   S[k,b] = sum_{i in batch b} R[k,i]           (K x B)
        #   Zsum[k,b,:] = sum_{i in batch b} R[k,i]*Z[i] (K x B x F)
        S = np.zeros((K, B), dtype=np.float64)
        Zsum = np.zeros((K, B, n_features), dtype=np.float64)

        for b in range(B):
            idx = self._batch_indices[b]
            if idx.size == 0:
                continue
            Rb = R[:, idx]  # (K x Nb)
            S[:, b] = Rb.sum(axis=1)
            Zsum[:, b, :] = Rb @ Z[idx, :]  # (K x F)

        # Prepare per-batch effect matrices E_b (K x F), with E_0 = 0 (reference batch)
        E_by_batch = np.zeros((B, K, n_features), dtype=np.float64)

        # Ridge penalty: apply only to batch coefficients (not intercept) to reduce bias
        lamb = float(self.lamb)
        ridge_diag = np.zeros(p, dtype=np.float64)
        if p > 1:
            ridge_diag[1:] = lamb

        eps = 1e-8

        # Build and solve small systems per cluster
        for k in range(K):
            S_k = S[k, :]  # (B,)
            s_k = float(S_k.sum())
            if s_k < min_cluster_mass:
                continue

            # Build A (p x p) and C (p x F) using structured design
            A = np.zeros((p, p), dtype=np.float64)
            A[0, 0] = s_k

            if B > 1:
                # A[0,j]=A[j,0]=S_k[j_batch], A[j,j]=S_k[j_batch], off-diagonal among batches is 0
                A[0, 1:] = S_k[1:]
                A[1:, 0] = S_k[1:]
                A[1:, 1:] = np.diag(S_k[1:])

            # Add ridge (batch terms only)
            if lamb > 0:
                A[np.diag_indices_from(A)] += ridge_diag

            C = np.zeros((p, n_features), dtype=np.float64)
            C[0, :] = Zsum[k, :, :].sum(axis=0)
            if B > 1:
                C[1:, :] = Zsum[k, 1:, :]

            # Solve A * beta = C
            if cho_factor is not None and cho_solve is not None:
                try:
                    c, low = cho_factor(A, lower=True, check_finite=False)
                    beta = cho_solve((c, low), C, check_finite=False)  # (p x F)
                except Exception:
                    beta = np.linalg.solve(A, C)
            else:
                beta = np.linalg.solve(A, C)

            # Batch effects (reference batch 0 is zero; other batches use beta[1:])
            if B > 1:
                E_by_batch[1:, k, :] = beta[1:, :]

        # Apply correction without per-cell loops:
        # delta[idx_b] += R[:, idx_b].T @ E_b  where E_b is (K x F)
        delta = np.zeros_like(Z, dtype=np.float64)
        for b in range(B):
            idx = self._batch_indices[b]
            if idx.size == 0:
                continue
            Eb = E_by_batch[b, :, :]  # (K x F)
            if np.max(np.abs(Eb)) < eps:
                continue
            delta[idx, :] += R[:, idx].T @ Eb

        alpha = float(self._correction_alpha)
        self.Z_corr = (self.Z_corr - alpha * delta).astype(self.Z_corr.dtype, copy=False)

        # Update cached norms after correction
        self._Z_sq = np.sum(self.Z_corr ** 2, axis=1)

        # Distances computed before correction are now stale; force recompute for objective
        self._last_dist = None

        # Return mean per-cell applied delta norm for early stopping / alpha adaptation
        mean_applied_delta = float(alpha * np.mean(np.linalg.norm(delta, axis=1)))
        return mean_applied_delta

    def _compute_objective(self) -> float:
        """Compute the Harmony objective function."""
        # Clustering objective (within-cluster variance)
        dist = self._last_dist if self._last_dist is not None else self._compute_distances()
        cluster_obj = np.sum(self.R * dist)

        # Diversity objective (entropy of batch distribution per cluster)
        expected = self.batch_props[np.newaxis, :]

        # Reuse cached O if available (computed during _update_R for current iteration)
        O = self._O_last
        if O is None:
            R_sum = self.R.sum(axis=1, keepdims=True) + 1e-8
            B = self._n_batches if self._n_batches is not None else self.Phi.shape[0]
            O = np.zeros((self.n_clusters, B), dtype=self.R.dtype)
            for b in range(B):
                idx = self._batch_indices[b] if self._batch_indices is not None else np.where(self._batch_id == b)[0]
                if idx.size == 0:
                    continue
                O[:, b] = self.R[:, idx].sum(axis=1) / R_sum[:, 0]

        theta_eff = float(getattr(self, "_theta_eff", self.theta))
        diversity_obj = theta_eff * np.sum(
            O * np.log((O + 1e-8) / (expected + 1e-8))
        )

        return cluster_obj + diversity_obj

    def transform(self, X: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted model.

        Args:
            X: New data matrix (n_cells x n_features)
            batch_labels: Batch labels for new cells

        Returns:
            Corrected coordinates
        """
        # This is a simplified transform - in practice would need more work
        return X


def run_harmony(
    X: np.ndarray,
    batch_labels: np.ndarray,
    n_clusters: int = 100,
    theta: float = 2.0,
    sigma: float = 0.1,
    lamb: float = 1.0,
    max_iter: int = 10,
    random_state: Optional[int] = None,
) -> Harmony:
    """
    Run Harmony algorithm.

    Args:
        X: Data matrix (n_cells x n_features), typically PCA coordinates
        batch_labels: Batch labels for each cell
        n_clusters: Number of clusters
        theta: Diversity penalty parameter
        sigma: Soft clustering width
        lamb: Ridge regression penalty
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Fitted Harmony object with Z_corr attribute containing corrected data

    Example:
        >>> X = np.random.randn(1000, 50)  # 1000 cells, 50 PCs
        >>> batch = np.repeat([0, 1, 2], [300, 400, 300])
        >>> hm = run_harmony(X, batch)
        >>> X_corrected = hm.Z_corr
    """
    hm = Harmony(
        n_clusters=n_clusters,
        theta=theta,
        sigma=sigma,
        lamb=lamb,
        max_iter=max_iter,
        random_state=random_state,
    )
    hm.fit(X, batch_labels)
    return hm
