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
from typing import Optional


class Harmony:
    """
    Harmony algorithm for batch effect correction.

    Internal shape conventions (enforced):
        Z_orig: (n_cells, n_features)
        Z_corr: (n_cells, n_features)
        Y:      (n_clusters, n_features) centroids
        R:      (n_clusters, n_cells)   soft assignments
        Phi:    (n_cells, n_batches)    one-hot batch membership

    Attributes:
        Z_corr: Corrected embedding after harmonization
        Z_orig: Original embedding
        R: Soft cluster assignments (clusters x cells)
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

        # Numerical constants / caches
        self._eps = 1e-12  # centralized epsilon for logs/divisions; keep consistent across objective/updates

        # Internal caches/state (initialized for clarity)
        self._unique_batches = None
        self._batch_index = None
        self._design = None
        self.batch_props = None
        self._Z_sq = None
        self._inv_sigma = None
        self._log_expected = None
        self._ridge_I = None

        # Reference batch policy (avoid input-order dependence): use largest batch as reference
        self._ref_batch = None
        self._ref_batch_idx = None

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
        X = np.asarray(X)
        batch_labels = np.asarray(batch_labels)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_cells, n_features).")
        n_cells, n_features = X.shape
        if batch_labels.ndim != 1 or batch_labels.shape[0] != n_cells:
            raise ValueError("batch_labels must be a 1D array with length n_cells.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/Inf values.")
        if self.n_clusters <= 0 or self.n_clusters > n_cells:
            raise ValueError("n_clusters must be in [1, n_cells].")
        if self.theta < 0:
            raise ValueError("theta must be >= 0.")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if self.lamb < 0:
            raise ValueError("lamb must be >= 0.")
        if self.max_iter <= 0 or self.max_iter_kmeans <= 0:
            raise ValueError("max_iter and max_iter_kmeans must be > 0.")
        if self.epsilon_cluster <= 0 or self.epsilon_harmony <= 0:
            raise ValueError("epsilon_cluster and epsilon_harmony must be > 0.")

        # Store original/corrected (float for stability)
        self.Z_orig = X.astype(np.float64, copy=True)
        self.Z_corr = self.Z_orig.copy()

        # Validate batch labels for predictable factorization.
        if batch_labels.dtype == object:
            # Reject None explicitly; NaN handling for object arrays is ambiguous.
            if any(b is None for b in batch_labels.tolist()):
                raise ValueError("batch_labels contains None; please provide valid batch identifiers.")
        else:
            if not np.isfinite(batch_labels).all():
                raise ValueError("batch_labels contains NaN/Inf values.")

        # Factorize batch labels efficiently and deterministically:
        # - stable first-occurrence order for unique batches
        # - no Python-level per-element dict mapping (faster for large n_cells)
        unique_raw, inv_raw, counts_raw = np.unique(
            batch_labels, return_inverse=True, return_counts=True
        )

        # Remap to stable first-occurrence order using argsort on first index.
        # Compute first occurrence per unique_raw id without Python loops.
        first_idx = np.full(unique_raw.shape[0], n_cells, dtype=np.int64)
        np.minimum.at(first_idx, inv_raw, np.arange(n_cells, dtype=np.int64))
        order = np.argsort(first_idx, kind="stable")
        unique_batches = unique_raw[order]
        counts = counts_raw[order]

        # Build inverse mapping: inv_stable = remap[inv_raw]
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size, dtype=order.dtype)
        inv = remap[inv_raw].astype(np.int64, copy=False)

        n_batches = unique_batches.shape[0]
        self._unique_batches = unique_batches
        self._batch_index = inv

        Phi = np.zeros((n_cells, n_batches), dtype=np.float64)
        Phi[np.arange(n_cells), inv] = 1.0
        self.Phi = Phi  # (n_cells x n_batches)

        # Batch proportions (n_batches,)
        self.batch_props = counts.astype(np.float64) / float(n_cells)

        # Cache expected proportions/log for diversity penalty
        self._log_expected = np.log(self.batch_props + self._eps)[np.newaxis, :]  # (1, n_batches)

        # Choose reference batch as largest batch to reduce order dependence.
        self._ref_batch_idx = int(np.argmax(counts))
        self._ref_batch = unique_batches[self._ref_batch_idx]

        # Cache commonly used design matrix for correction: intercept + (all but ref batch)
        # Shape: (n_cells, n_cov) where n_cov = 1 + (n_batches-1)
        if n_batches > 1:
            mask = np.ones(n_batches, dtype=bool)
            mask[self._ref_batch_idx] = False
            Phi_no_ref = self.Phi[:, mask]
            self._design = np.concatenate(
                (np.ones((n_cells, 1), dtype=np.float64), Phi_no_ref),
                axis=1,
            )
        else:
            # Single batch: only intercept (correction becomes no-op)
            self._design = np.ones((n_cells, 1), dtype=np.float64)

        # Precompute constants/caches
        self._inv_sigma = 1.0 / float(self.sigma)

        n_cov = self._design.shape[1]
        self._ridge_I = np.eye(n_cov, dtype=np.float64)
        self._ridge_I[0, 0] = 0.0  # do not penalize intercept

        # Precompute batch index lists once for correction/objective
        # (kept as a list-of-arrays to avoid building big boolean masks)
        self._batch_cell_idx = [np.flatnonzero(inv == b) for b in range(n_batches)]

        # Precompute for distance calcs
        self._Z_sq = np.sum(self.Z_corr * self.Z_corr, axis=1, keepdims=True)  # (n_cells, 1)

        # Initialize clusters
        self._init_clusters()

        # Main Harmony loop
        self.objectives = []
        for iteration in range(self.max_iter):
            self._cluster()
            self._correct()

            obj = self._compute_objective()
            self.objectives.append(obj)

            if iteration > 0:
                prev = self.objectives[-2]
                # Relative+absolute stopping improves stability across scales
                denom = max(1.0, abs(prev))
                if abs(prev - obj) / denom < self.epsilon_harmony:
                    break

        return self

    def _init_clusters(self):
        """Initialize cluster centroids using k-means."""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=1,
            max_iter=25,
            algorithm="lloyd",
        )
        kmeans.fit(self.Z_corr)
        # Store centroids as (n_clusters x n_features) for simpler/cheaper distance math
        self.Y = kmeans.cluster_centers_.astype(np.float64, copy=False)

        # Initialize soft assignments
        self._update_R()

    def _cluster(self):
        """Run clustering iterations."""
        n_cells = self.Z_corr.shape[0]
        for _ in range(self.max_iter_kmeans):
            # Update centroids
            self._update_centroids()

            # Update soft assignments
            R_old = self.R.copy() if self.R is not None else None
            self._update_R()

            # Enforce shapes to prevent silent axis/orientation bugs
            if self.R.shape != (self.n_clusters, n_cells):
                raise RuntimeError(f"R has wrong shape {self.R.shape}, expected {(self.n_clusters, n_cells)}")

            # Check convergence
            if R_old is not None:
                r_change = float(np.max(np.abs(self.R - R_old)))
                if r_change < self.epsilon_cluster:
                    break

    def _update_centroids(self):
        """Update cluster centroids."""
        # R: (n_clusters x n_cells), Z: (n_cells x n_features)
        weights = self.R
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8  # (n_clusters, 1)

        # Y[k] = sum_i R[k,i] * Z[i] / sum_i R[k,i]
        self.Y = (weights @ self.Z_corr) / weights_sum  # (n_clusters x n_features)

    def _update_R(self):
        """Update soft cluster assignments with diversity penalty.

        Performance/stability:
        - Use log-sum-exp normalization across clusters for numerical stability
        - Apply diversity penalty in log-space and normalize once
        """
        # dist: (n_clusters x n_cells)
        dist = self._compute_distances()

        # Base log-probabilities (up to a per-cell constant)
        logR = (-dist * self._inv_sigma).clip(min=-60.0, max=60.0)  # (K, N)

        if self.theta > 0 and self.Phi.shape[1] > 1:
            # Compute O using stabilized R derived from logR
            # (need actual R for the batch composition O = (R Phi) / sum R)
            m = np.max(logR, axis=0, keepdims=True)
            R_tmp = np.exp(logR - m)
            R_sum = R_tmp.sum(axis=1, keepdims=True) + self._eps
            O = (R_tmp @ self.Phi) / R_sum  # (K, B)

            # KL(O || expected)
            logO = np.log(O + self._eps)
            penalty = self.theta * np.sum(O * (logO - self._log_expected), axis=1, keepdims=True)  # (K,1)

            # Apply cluster-wide penalty in log-space
            logR = logR - penalty

        # Normalize over clusters per cell using log-sum-exp
        m = np.max(logR, axis=0, keepdims=True)
        R = np.exp(logR - m)
        R /= (R.sum(axis=0, keepdims=True) + self._eps)

        self.R = R

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids.

        Returns:
            dist: (n_clusters x n_cells) where dist[k,i] = ||Z[i] - Y[k]||^2
        """
        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 z¡¤y
        # Use explicit temporaries to help BLAS and avoid extra transposes.
        Z = self.Z_corr
        Y = self.Y
        Z_sq = self._Z_sq  # (n_cells, 1)
        Y_sq = np.sum(Y * Y, axis=1, keepdims=True).T  # (1, n_clusters)
        cross = Z @ Y.T  # (n_cells, n_clusters)
        dist = Z_sq + Y_sq
        dist -= 2.0 * cross
        # small negative values can arise from roundoff; keep objective stable
        np.maximum(dist, 0.0, out=dist)
        return dist.T

    def _correct(self):
        """Apply linear correction to remove batch effects.

        Optimizations:
        - Use precomputed per-batch cell indices (no per-iteration np.flatnonzero)
        - Avoid forming wz = w[:,None]*Z for every cluster (huge); use Z.T @ w and per-batch sums
        - Use analytic X^T W X for intercept + one-hot dummies (dummies disjoint)
        """
        n_cov = self._design.shape[1]
        if n_cov == 1:
            return

        design = self._design
        ridge_I = self._ridge_I
        batch_cell_idx = self._batch_cell_idx
        Z = self.Z_corr
        ZT = Z.T
        n_batches = len(batch_cell_idx)
        # n_cov = 1 + (n_batches-1)
        n_features = Z.shape[1]

        # Indices for non-reference batches in the same order as design[:, 1:]
        if n_batches > 1:
            nonref_batches = [b for b in range(n_batches) if b != self._ref_batch_idx]
        else:
            nonref_batches = []

        for k in range(self.n_clusters):
            w = self.R[k, :].astype(np.float64, copy=False)  # (n_cells,)
            sw = float(w.sum())
            if sw < 1e-10:
                continue

            # Weighted sums per group (intercept + each non-ref batch)
            s = np.empty(n_cov, dtype=np.float64)
            s[0] = sw
            # Also compute X^T W Z rows without building wz
            XWZ = np.empty((n_cov, n_features), dtype=np.float64)
            # intercept row: sum_i w_i * z_i
            XWZ[0, :] = (ZT @ w)

            for j, b in enumerate(nonref_batches, start=1):
                idx = batch_cell_idx[b]
                if idx.size:
                    wb = w[idx]
                    s[j] = float(wb.sum())
                    # sum_{i in batch b} w_i * z_i
                    XWZ[j, :] = (Z[idx, :].T @ wb)
                else:
                    s[j] = 0.0
                    XWZ[j, :] = 0.0

            # Build X^T W X
            XWX = np.zeros((n_cov, n_cov), dtype=np.float64)
            XWX[0, 0] = s[0]
            XWX[0, 1:] = s[1:]
            XWX[1:, 0] = s[1:]
            XWX[np.arange(1, n_cov), np.arange(1, n_cov)] = s[1:]

            if self.lamb > 0:
                XWX = XWX + self.lamb * ridge_I

            # Solve for beta
            try:
                L = np.linalg.cholesky(XWX)
                tmp = np.linalg.solve(L, XWZ)
                beta = np.linalg.solve(L.T, tmp)
            except np.linalg.LinAlgError:
                try:
                    beta = np.linalg.solve(XWX, XWZ)
                except np.linalg.LinAlgError:
                    continue

            # Remove batch effects only (exclude intercept)
            batch_effect = design[:, 1:] @ beta[1:, :]  # (n_cells, n_features)

            # Apply correction weighted by cluster membership
            Z -= w[:, None] * batch_effect

        self.Z_corr = Z
        # Update cache for distance computations
        self._Z_sq = np.sum(self.Z_corr * self.Z_corr, axis=1, keepdims=True)

    def _compute_objective(self) -> float:
        """Compute the Harmony objective function."""
        dist = self._compute_distances()
        cluster_obj = float(np.sum(self.R * dist))

        diversity_obj = 0.0
        if self.theta > 0 and self.Phi.shape[1] > 1:
            R_sum = self.R.sum(axis=1, keepdims=True) + self._eps
            O = (self.R @ self.Phi) / R_sum
            logO = np.log(O + self._eps)
            diversity_obj = float(self.theta * np.sum(O * (logO - self._log_expected)))

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
