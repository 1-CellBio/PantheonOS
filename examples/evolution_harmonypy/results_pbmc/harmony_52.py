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
from typing import Optional, List

try:
    from scipy.linalg import cho_factor, cho_solve
except Exception:  # pragma: no cover
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
        self.Phi_reduced = None  # (n_batches-1 x n_cells)
        self.design = None  # (n_cells x n_batches) = [1, Phi_reduced.T]
        self.objectives = []

        # Cached shapes
        self.n_cells = None
        self.n_features = None
        self.n_batches = None

        # Batch indexing (set in fit)
        self.batch_id = None  # (n_cells,) int in [0..n_batches-1]
        self.batch_indices: Optional[List[np.ndarray]] = None  # list of indices per batch

        # Cached batch-weight stats for current R
        self._S = None  # (n_clusters x n_batches) = R @ Phi.T
        self._R_sum = None  # (n_clusters x 1) = sum over cells

        # Correction damping (step size)
        self._alpha = 0.7

        # Distance cache (versioned; invalidated when Z_corr or Y changes)
        self._Z_version = 0
        self._Y_version = 0
        self._dist_cache = None
        self._dist_cache_key = None  # (_Z_version, _Y_version)
        self._Z_sq_cache = None
        self._Z_sq_cache_key = None  # _Z_version
        self._Y_sq_cache = None
        self._Y_sq_cache_key = None  # _Y_version

        # Sigma reference (EMA) for smoother/stabler soft assignments
        self._sigma_ref = None

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
        self.n_cells, self.n_features = n_cells, n_features

        # Store original (use float32 for large matrices)
        X32 = np.asarray(X, dtype=np.float32)
        self.Z_orig = X32.copy()
        self.Z_corr = X32.copy()

        # Create batch membership matrix (one-hot encoding) + fast batch indices
        unique_batches, batch_id = np.unique(batch_labels, return_inverse=True)
        batch_id = np.asarray(batch_id, dtype=np.int32)
        self.batch_id = batch_id
        n_batches = len(unique_batches)
        self.n_batches = n_batches

        self.batch_indices = [np.where(batch_id == b)[0] for b in range(n_batches)]

        # Keep Phi compact; cast to float only when needed in math
        self.Phi = np.zeros((n_batches, n_cells), dtype=np.float32)
        for b in range(n_batches):
            self.Phi[b, self.batch_indices[b]] = 1.0

        # Precompute reduced Phi and design matrix (constant across clusters/iterations)
        self.Phi_reduced = self.Phi[1:, :]  # (n_batches-1 x n_cells)
        self.design = np.vstack([np.ones(n_cells, dtype=np.float32), self.Phi_reduced]).T  # (n_cells x n_batches)

        # Compute batch proportions
        self.batch_props = self.Phi.sum(axis=1) / n_cells

        # Initialize clusters
        self._init_clusters()

        # Main Harmony loop (with light scheduling for theta/alpha)
        self.objectives = []
        self._alpha = float(getattr(self, "_alpha", 0.7))
        for iteration in range(self.max_iter):
            # Theta warm start + mild decay toward base theta
            if iteration < 2:
                self._theta_t = float(self.theta) * 1.25
            else:
                # smooth decay toward theta (or slightly below) to avoid over-mixing
                self._theta_t = float(self.theta) + 0.25 * float(self.theta) * np.exp(-0.5 * (iteration - 2))

            # Clustering step
            dist = self._cluster()

            # Correction step (simultaneous update)
            self._correct()

            # Check convergence (reuse latest distances if available; otherwise compute once)
            if dist is None:
                dist = self._compute_distances()
            obj = self._compute_objective(dist=dist)
            self.objectives.append(obj)

            # Adaptive damping based on objective behavior (protect bio structure / reduce oscillation)
            if iteration > 0:
                prev = self.objectives[-2]
                cur = self.objectives[-1]
                obj_change = abs(prev - cur)

                if cur > prev + 1e-8:
                    # objective increased -> reduce step size and ease mixing pressure
                    self._alpha = max(0.3, float(self._alpha) * 0.7)
                    self._theta_t = max(0.0, float(self._theta_t) * 0.95)
                else:
                    # objective decreased -> cautiously raise alpha
                    self._alpha = min(1.0, float(self._alpha) * 1.05)

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
        self.Y = kmeans.cluster_centers_.astype(np.float32).T  # (n_features x n_clusters)

        # Y changed
        self._Y_version += 1
        self._dist_cache = None
        self._dist_cache_key = None
        self._Y_sq_cache = None
        self._Y_sq_cache_key = None

        # Initialize soft assignments
        self._update_R()

    def _cluster(self):
        """Run clustering iterations.

        Returns:
            Latest distance matrix (n_clusters x n_cells) from the final update.
        """
        dist = None
        R_prev = None
        for _ in range(self.max_iter_kmeans):
            # Update centroids
            self._update_centroids()

            # Update soft assignments
            if self.R is not None:
                if R_prev is None or R_prev.shape != self.R.shape:
                    R_prev = np.empty_like(self.R)
                R_prev[...] = self.R
            dist = self._compute_distances()
            self._update_R(dist=dist)

            # Check convergence
            if R_prev is not None:
                r_change = np.max(np.abs(self.R - R_prev))
                if r_change < self.epsilon_cluster:
                    break
        return dist

    def _update_centroids(self):
        """Update cluster centroids."""
        # Weighted average of cells
        weights = self.R  # (n_clusters x n_cells)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8

        # Y = Z @ R.T / sum(R)
        self.Y = (self.Z_corr.T @ weights.T) / weights_sum.T

        # Invalidate distance cache (Y changed)
        self._Y_version += 1
        self._dist_cache = None
        self._dist_cache_key = None
        self._Y_sq_cache = None
        self._Y_sq_cache_key = None

    def _update_R(self, dist: Optional[np.ndarray] = None):
        """Update soft cluster assignments with diversity penalty.

        Optimizations:
        - Compute soft assignments with at most two softmax passes (one if theta==0)
        - Replace dense R @ Phi.T with cheap batch-wise reductions using batch indices

        Args:
            dist: Optional precomputed squared distances (n_clusters x n_cells)
        """
        if dist is None:
            dist = self._compute_distances()

        # Cheaper, smoother scale estimate (avoid per-cell median over K)
        sigma_ref = float(np.mean(np.min(dist, axis=0)))
        if self._sigma_ref is None:
            self._sigma_ref = sigma_ref
        else:
            self._sigma_ref = 0.9 * float(self._sigma_ref) + 0.1 * sigma_ref

        sigma_scaled = max(self.sigma * float(self._sigma_ref), 1e-8)

        # Stable soft assignments in log-space
        logR = -dist / sigma_scaled

        # Use scheduled theta if present (keeps public API unchanged)
        theta_t = float(getattr(self, "_theta_t", self.theta))

        # First softmax (unpenalized)
        logR0 = logR - logR.max(axis=0, keepdims=True)
        R = np.exp(logR0)
        R = R / (R.sum(axis=0, keepdims=True) + 1e-8)

        if theta_t > 0:
            # Compute cluster-by-batch composition under current assignments via reductions
            n_batches = self.n_batches if self.n_batches is not None else self.Phi.shape[0]
            if self.batch_indices is None:
                self.batch_indices = [np.where(self.batch_id == b)[0] for b in range(n_batches)]

            S = np.zeros((self.n_clusters, n_batches), dtype=np.float32)
            for b in range(n_batches):
                idx = self.batch_indices[b]
                if idx.size == 0:
                    continue
                # sum over cells in batch b for each cluster
                S[:, b] = R[:, idx].sum(axis=1)

            R_sum = R.sum(axis=1, keepdims=True) + 1e-8
            O = S / R_sum  # (K x B)
            expected = self.batch_props[np.newaxis, :]  # (1 x B)
            log_ratio = np.log(O + 1e-8) - np.log(expected + 1e-8)  # (K x B)

            # Apply penalty to each cell by selecting its batch column (O(KN) gather; avoids KxN * N x B)
            penalty_kn = log_ratio[:, self.batch_id]  # (K x N)
            logR = logR - theta_t * penalty_kn

            # Second softmax (penalized)
            logR -= logR.max(axis=0, keepdims=True)
            R = np.exp(logR)
            R = R / (R.sum(axis=0, keepdims=True) + 1e-8)

            # Cache S and R_sum for final R as well (recompute cheaply)
            S = np.zeros((self.n_clusters, n_batches), dtype=np.float32)
            for b in range(n_batches):
                idx = self.batch_indices[b]
                if idx.size == 0:
                    continue
                S[:, b] = R[:, idx].sum(axis=1)
            self._S = S.astype(np.float32, copy=False)
            self._R_sum = R.sum(axis=1, keepdims=True).astype(np.float32, copy=False)
        else:
            # theta==0: reuse S from reductions (or compute once if needed)
            n_batches = self.n_batches if self.n_batches is not None else self.Phi.shape[0]
            if self.batch_indices is None:
                self.batch_indices = [np.where(self.batch_id == b)[0] for b in range(n_batches)]
            S = np.zeros((self.n_clusters, n_batches), dtype=np.float32)
            for b in range(n_batches):
                idx = self.batch_indices[b]
                if idx.size == 0:
                    continue
                S[:, b] = R[:, idx].sum(axis=1)
            self._S = S.astype(np.float32, copy=False)
            self._R_sum = R.sum(axis=1, keepdims=True).astype(np.float32, copy=False)

        self.R = R.astype(np.float32, copy=False)

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids (versioned cache + cached norms)."""
        key = (self._Z_version, self._Y_version)
        if self._dist_cache is not None and self._dist_cache_key == key:
            return self._dist_cache

        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 * z @ y
        if self._Z_sq_cache is None or self._Z_sq_cache_key != self._Z_version:
            self._Z_sq_cache = np.sum(self.Z_corr ** 2, axis=1, keepdims=True)  # (n_cells x 1)
            self._Z_sq_cache_key = self._Z_version
        Z_sq = self._Z_sq_cache

        if self._Y_sq_cache is None or self._Y_sq_cache_key != self._Y_version:
            self._Y_sq_cache = np.sum(self.Y ** 2, axis=0, keepdims=True)  # (1 x n_clusters)
            self._Y_sq_cache_key = self._Y_version
        Y_sq = self._Y_sq_cache

        cross = self.Z_corr @ self.Y  # (n_cells x n_clusters)

        dist = (Z_sq + Y_sq - 2.0 * cross).T.astype(np.float32, copy=False)  # (n_clusters x n_cells)
        self._dist_cache = dist
        self._dist_cache_key = key
        return dist

    def _correct(self):
        """Apply linear correction to remove batch effects (simultaneous update).

        Optimizations:
        - Avoid building a (K, B, F) tensor (Zw). Compute only XWZ0 (K,F) and per-batch XWZ1 (B-1,K,F).
        - Replace per-cluster Cholesky with a closed-form solver exploiting XWX block structure.
        - Apply corrections batch-by-batch using one GEMM per batch: delta[idx] += Rb.T @ Beta_batch[b]
        - Dampen correction step via self._alpha (scheduled in fit()).
        """
        n_cells = self.n_cells if self.n_cells is not None else self.Z_corr.shape[0]
        n_batches = self.n_batches if self.n_batches is not None else self.Phi.shape[0]
        eps = 1e-8

        if n_batches <= 1:
            return

        # Snapshot base to avoid order-dependent in-loop updates
        Z_base = self.Z_corr.copy()

        if self.batch_indices is None:
            self.batch_indices = [np.where(self.batch_id == b)[0] for b in range(n_batches)]

        # Precompute wb (K x B) and XWZ parts:
        # - XWZ0: (K x F) = sum_b (R[:,idx_b] @ Zb)
        # - XWZ1_list[b-1]: (K x F) for b in 1..B-1
        wb = np.zeros((self.n_clusters, n_batches), dtype=np.float32)
        XWZ0 = np.zeros((self.n_clusters, self.n_features), dtype=np.float32)
        XWZ1_list = [np.zeros((self.n_clusters, self.n_features), dtype=np.float32) for _ in range(n_batches - 1)]

        for b in range(n_batches):
            idx = self.batch_indices[b]
            if idx.size == 0:
                continue
            Rb = self.R[:, idx]  # (K x n_b)
            Zb = Z_base[idx, :]  # (n_b x F)
            wb[:, b] = Rb.sum(axis=1)
            M = (Rb @ Zb).astype(np.float32, copy=False)  # (K x F)
            XWZ0 += M
            if b > 0:
                XWZ1_list[b - 1][...] = M

        lamb = float(self.lamb)
        if lamb <= 0:
            # Ensure stability for closed-form solver
            lamb = 1e-6

        # Store per-batch cluster effects Beta_batch[b-1] = beta_k[b,:] stacked over k -> (K x F)
        Beta_batch = [np.zeros((self.n_clusters, self.n_features), dtype=np.float32) for _ in range(n_batches - 1)]

        # Closed-form solve per cluster exploiting:
        # XWX = [[a, v^T],
        #        [v,  D ]]
        # where a = w0 + λ, v = wb_k[1:], D = diag(v + λ)
        for k in range(self.n_clusters):
            wb_k = wb[k, :]  # (B,)
            w0 = float(wb_k.sum())
            if w0 < eps:
                continue

            v = wb_k[1:].astype(np.float32, copy=False)  # (B-1,)
            a = w0 + lamb
            invD = 1.0 / (v + lamb + eps)  # (B-1,)

            # Schur complement scalar
            s = a - float(np.sum((v * v) * invD))
            if s <= eps:
                continue

            XWZ0_k = XWZ0[k, :].astype(np.float32, copy=False)  # (F,)
            # XWZ1: (B-1 x F)
            XWZ1_k = np.stack([XWZ1_list[j][k, :] for j in range(n_batches - 1)], axis=0).astype(np.float32, copy=False)

            # beta0: (F,)
            t = (invD[:, None] * XWZ1_k)  # (B-1 x F)
            beta0 = (XWZ0_k - np.sum(v[:, None] * t, axis=0)) / (s + eps)

            # beta1: (B-1 x F)
            beta1 = invD[:, None] * (XWZ1_k - v[:, None] * beta0[None, :])

            # Save into per-batch structure
            for j in range(n_batches - 1):
                Beta_batch[j][k, :] = beta1[j, :]

        # Apply corrections batch-by-batch using GEMM:
        # delta_sum[idx] += Rb.T @ Beta_batch[b-1]
        delta_sum = np.zeros_like(self.Z_corr, dtype=np.float32)
        w_tot = np.zeros(n_cells, dtype=np.float32)

        for b in range(1, n_batches):
            idx = self.batch_indices[b]
            if idx.size == 0:
                continue
            Rb = self.R[:, idx]  # (K x n_b)
            delta_sum[idx, :] += (Rb.T @ Beta_batch[b - 1]).astype(np.float32, copy=False)  # (n_b x F)
            w_tot[idx] += Rb.sum(axis=0).astype(np.float32, copy=False)

        raw_delta = delta_sum / (w_tot[:, None] + eps)
        alpha = float(getattr(self, "_alpha", 1.0))
        self.Z_corr = (Z_base - alpha * raw_delta).astype(np.float32, copy=False)

        # Invalidate distance/norm caches (Z_corr changed)
        self._Z_version += 1
        self._dist_cache = None
        self._dist_cache_key = None
        self._Z_sq_cache = None
        self._Z_sq_cache_key = None

    def _compute_objective(self, dist: Optional[np.ndarray] = None) -> float:
        """Compute the Harmony objective function.

        Args:
            dist: Optional precomputed squared distances (n_clusters x n_cells)
        """
        # Clustering objective (within-cluster variance)
        if dist is None:
            dist = self._compute_distances()
        cluster_obj = np.sum(self.R * dist)

        # Diversity objective (entropy of batch distribution per cluster)
        R_sum = (self._R_sum if self._R_sum is not None else self.R.sum(axis=1, keepdims=True)).astype(np.float32, copy=False) + 1e-8
        S = self._S if self._S is not None else (self.R @ self.Phi.T)
        O = S / R_sum
        expected = self.batch_props[np.newaxis, :]
        diversity_obj = self.theta * np.sum(
            O * (np.log(O + 1e-8) - np.log(expected + 1e-8))
        )

        return float(cluster_obj + diversity_obj)

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
