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

        # Reusable distance computation buffers (avoid KxN alloc + transpose copies)
        self._cross_buf = None  # (n_cells x n_clusters) float32
        self._dist_buf = None  # (n_clusters x n_cells) float32

        # Cache R.T to avoid repeated (N x K) copies in _correct()
        self._R_version = 0
        self._R_T_cache = None
        self._R_T_cache_key = None  # _R_version

        # Per-feature correction clipping cache/state
        self._feat_clip_outer_iter = -1
        self._feat_cap = None  # (n_features,) float32

        # Sigma reference (EMA) for smoother/stabler soft assignments
        self._sigma_ref = None
        self._sigma_ref_outer_iter = -1  # update sigma_ref once per outer Harmony iter

        # Persistent RNG (varied subsamples across iterations)
        self._rng = np.random.default_rng(self.random_state)

        # Cached per-cluster divergence (used for cluster-aware correction gating)
        self._kl_k = None

        # Cache subsample indices for per-feature caps once per outer iteration
        self._feat_cap_sub_idx_outer_iter = -1
        self._feat_cap_sub_idx = None

        # Optional sorted-space buffer for _update_R diversity penalty (Change 5)
        self._logR_sorted_buf = None

        # Persistent gather buffers to avoid per-iteration allocations (Change A/C)
        self._Z_sorted_buf = None          # (n_cells x n_features) float32
        self._R_T_sorted_buf = None        # (n_cells x n_clusters) float32
        self._R_sorted_buf = None          # (n_clusters x n_cells) float32

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

        # Store original (use float32 for large matrices) and ensure contiguous for BLAS
        X32 = np.asarray(X, dtype=np.float32)
        self.Z_orig = np.ascontiguousarray(X32.copy())
        self.Z_corr = np.ascontiguousarray(X32.copy())

        # Create batch indexing (primary internal representation)
        unique_batches, batch_id = np.unique(batch_labels, return_inverse=True)
        batch_id = np.asarray(batch_id, dtype=np.int32)
        n_batches = len(unique_batches)
        self.n_batches = n_batches

        # Choose reference batch as the largest batch, and remap so reference becomes batch 0.
        # This stabilizes baseline and reduces biased correction.
        batch_counts_int = np.bincount(batch_id, minlength=n_batches).astype(np.int64, copy=False)
        ref_old = int(np.argmax(batch_counts_int))
        perm = np.arange(n_batches, dtype=np.int32)
        if ref_old != 0:
            perm[0], perm[ref_old] = perm[ref_old], perm[0]
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n_batches, dtype=np.int32)

        self._batch_perm = perm          # old -> new via inv_perm[old]
        self._batch_inv_perm = inv_perm  # new -> old via perm[new]
        self.ref_batch_old = ref_old
        self.ref_batch = 0

        self.batch_id = inv_perm[batch_id]
        batch_id = self.batch_id

        # Precompute per-batch indices and proportions (in remapped order)
        self.batch_indices = [np.where(batch_id == b)[0] for b in range(n_batches)]
        batch_counts = np.array([idx.size for idx in self.batch_indices], dtype=np.float32)
        self.batch_props = batch_counts / float(n_cells)

        # Precompute a contiguous "cells sorted by batch" layout for fast slicing in hot paths
        # order: indices of cells sorted by batch_id (stable to preserve original order within batch)
        self._order = np.argsort(self.batch_id, kind="stable").astype(np.int64, copy=False)
        # inv_order: inverse permutation such that inv_order[order[i]] = i
        self._inv_order = np.empty_like(self._order)
        self._inv_order[self._order] = np.arange(n_cells, dtype=self._order.dtype)

        # Permanently reorder internal arrays into sorted-by-batch space (Change 1).
        # From this point onward, the internal order is canonical and batches are contiguous blocks.
        Z_orig_sorted = np.empty_like(self.Z_orig)
        Z_corr_sorted = np.empty_like(self.Z_corr)
        np.take(self.Z_orig, self._order, axis=0, out=Z_orig_sorted)
        np.take(self.Z_corr, self._order, axis=0, out=Z_corr_sorted)
        self.Z_orig = np.ascontiguousarray(Z_orig_sorted)
        self.Z_corr = np.ascontiguousarray(Z_corr_sorted)

        self.batch_id = np.ascontiguousarray(self.batch_id[self._order])

        # Now that batch_id is sorted, batch_ptr becomes canonical (Change 1).
        batch_counts_int2 = np.bincount(self.batch_id, minlength=n_batches).astype(np.int64, copy=False)
        self._batch_ptr = np.zeros(n_batches + 1, dtype=np.int64)
        self._batch_ptr[1:] = np.cumsum(batch_counts_int2)

        # In sorted space we don't need batch_indices for hot paths anymore.
        self.batch_indices = None

        # Precompute expected/log_expected once (Change 4)
        self._expected = self.batch_props[np.newaxis, :].astype(np.float32, copy=False)
        self._log_expected = np.log(self._expected + 1e-8).astype(np.float32, copy=False)

        # Keep Phi optional for API/debugging; avoid using it in hot loops.
        # If you need Phi externally, you can uncomment the following block to materialize it.
        self.Phi = None
        self.Phi_reduced = None
        self.design = None

        # Initialize clusters
        self._init_clusters()

        # Main Harmony loop
        self.objectives = []
        # Theta scheduling state
        self._theta_t = float(self.theta)
        self._theta_decay = 1.0
        for iteration in range(self.max_iter):
            # Mark current outer iteration for sigma_ref update cadence
            self._outer_iter = int(iteration)

            # Theta schedule: warmup for first few iterations to encourage early mixing,
            # then allow mild decay when objective stalls/increases to protect bio structure.
            warmup = min(1.0, float(iteration + 1) / 3.0)
            self._theta_t = float(self.theta) * warmup * float(self._theta_decay)

            # Clustering step
            dist = self._cluster()

            # Correction step (simultaneous update)
            self._correct()

            # Check convergence (reuse latest distances if available; otherwise compute once)
            if dist is None:
                dist = self._compute_distances()
            obj = self._compute_objective(dist=dist)
            self.objectives.append(obj)

            # Adaptive damping based on objective progression (guardrail)
            if iteration > 0:
                prev = float(self.objectives[-2])
                curr = float(self.objectives[-1])
                if curr > prev + 1e-12:
                    # Objective increased => reduce step size to prevent over-correction/oscillation
                    self._alpha = max(0.1, float(getattr(self, "_alpha", 0.7)) * 0.5)
                    # Also taper theta slightly if we're not improving (helps prevent overmixing late)
                    self._theta_decay = max(0.5, float(self._theta_decay) * 0.9)
                else:
                    # Objective decreased => slowly relax back toward a max
                    self._alpha = min(0.9, float(getattr(self, "_alpha", 0.7)) * 1.05)
                    # Allow theta to recover a bit (bounded) if improving
                    self._theta_decay = min(1.0, float(self._theta_decay) * 1.02)

                obj_change = abs(prev - curr)
                # Early stopping: require both objective stabilization and small correction magnitude
                # (median correction norm is tracked inside _correct via self._last_median_delta)
                if obj_change < self.epsilon_harmony:
                    med = float(getattr(self, "_last_median_delta", np.inf))
                    # Threshold relative to embedding scale (robust)
                    emb = self.Z_corr
                    scale = float(np.percentile(np.linalg.norm(emb, axis=1), 50)) + 1e-8
                    if med < 1e-3 * scale:
                        break

        # Restore original cell order for public API (Change 1)
        Z_unsorted = np.empty_like(self.Z_corr)
        Z_unsorted[self._inv_order, :] = self.Z_corr
        self.Z_corr = Z_unsorted

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

        # Adaptive inner iterations late in Harmony when corrections are already small (Change F)
        max_iter_kmeans = int(self.max_iter_kmeans)
        med = float(getattr(self, "_last_median_delta", np.inf))
        if np.isfinite(med):
            emb = self.Z_corr
            scale = float(np.percentile(np.linalg.norm(emb, axis=1), 50)) + 1e-8
            if med < 1e-3 * scale:
                max_iter_kmeans = min(max_iter_kmeans, 5)
            else:
                outer_iter = int(getattr(self, "_outer_iter", 0))
                if outer_iter >= 3:
                    max_iter_kmeans = min(max_iter_kmeans, max(5, max_iter_kmeans // 2))

        # Centroid-movement early stop buffer
        F = self.n_features if self.n_features is not None else self.Z_corr.shape[1]
        K = self.n_clusters
        if getattr(self, "_Y_prev_buf", None) is None or self._Y_prev_buf.shape != (F, K):
            self._Y_prev_buf = np.empty((F, K), dtype=np.float32)

        # Threshold for centroid stabilization (relative to typical centroid norm)
        y_shift_eps = float(getattr(self, "_epsilon_centroid", 1e-3))

        for _ in range(max_iter_kmeans):
            # Snapshot previous centroids
            self._Y_prev_buf[...] = self.Y

            # Update centroids
            self._update_centroids()

            # Cheaper early stop: if centroids barely move, stop before distance/softmax
            dY = self.Y - self._Y_prev_buf
            num = float(np.max(np.linalg.norm(dY, axis=0)))
            denom = float(np.max(np.linalg.norm(self._Y_prev_buf, axis=0))) + 1e-8
            if (num / denom) < y_shift_eps:
                break

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
        """Update cluster centroids.

        Computes a proposed centroid update but only commits (and invalidates caches)
        if the change is non-trivial. This reduces cache churn late in k-means.
        """
        # Weighted average of cells
        weights = self.R  # (n_clusters x n_cells)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8

        # Proposed update: Y_new = Z @ R.T / sum(R)
        Y_new = (self.Z_corr.T @ weights.T) / weights_sum.T

        # Commit only if moved enough
        y_shift_eps = float(getattr(self, "_epsilon_centroid", 1e-3))
        dY = Y_new - self.Y
        num = float(np.max(np.linalg.norm(dY, axis=0)))
        denom = float(np.max(np.linalg.norm(self.Y, axis=0))) + 1e-8
        if (num / denom) < y_shift_eps:
            return

        self.Y = Y_new

        # Invalidate distance cache (Y changed)
        self._Y_version += 1
        self._dist_cache = None
        self._dist_cache_key = None
        self._Y_sq_cache = None
        self._Y_sq_cache_key = None

    def _update_R(self, dist: Optional[np.ndarray] = None):
        """Update soft cluster assignments with diversity penalty.

        Args:
            dist: Optional precomputed squared distances (n_clusters x n_cells)
        """
        if dist is None:
            dist = self._compute_distances()

        eps = 1e-8
        K = self.n_clusters
        N = self.n_cells if self.n_cells is not None else dist.shape[1]

        # Robust, smoother scale estimate (median of nearest-centroid distances)
        # Speed: use O(N) partition median instead of percentile.
        # Stability: update sigma_ref once per outer Harmony iteration to reduce jitter.
        outer_iter = int(getattr(self, "_outer_iter", -1))
        if self._sigma_ref is None or self._sigma_ref_outer_iter != outer_iter:
            nearest = np.min(dist, axis=0)
            mid = int(nearest.size // 2)
            sigma_ref = float(np.partition(nearest, mid)[mid])
            if self._sigma_ref is None:
                self._sigma_ref = sigma_ref
            else:
                self._sigma_ref = 0.9 * float(self._sigma_ref) + 0.1 * sigma_ref
            self._sigma_ref_outer_iter = outer_iter

        sigma_scaled = max(self.sigma * float(self._sigma_ref), eps)

        # Reuse buffers to avoid repeated KxN allocations
        if getattr(self, "_logR_buf", None) is None or self._logR_buf.shape != (K, N):
            self._logR_buf = np.empty((K, N), dtype=np.float32)
        logR = self._logR_buf

        # Stable soft assignments in log-space (in-place)
        logR[...] = (-dist / sigma_scaled)

        # Diversity penalty using cached composition from previous R (fast + stable)
        theta_t = float(getattr(self, "_theta_t", self.theta))
        if theta_t > 0:
            expected = getattr(self, "_expected", None)
            log_expected = getattr(self, "_log_expected", None)
            if expected is None or log_expected is None:
                expected = self.batch_props[np.newaxis, :].astype(np.float32, copy=False)  # (1 x B)
                log_expected = np.log(expected + eps).astype(np.float32, copy=False)

            if self._S is not None and self._R_sum is not None:
                O_prev = self._S / (self._R_sum + eps)  # (K x B)
                log_ratio = np.log(O_prev + eps) - log_expected  # (K x B)
            else:
                # First call fallback: no previous composition available
                O_prev = None
                log_ratio = np.zeros((K, self.n_batches), dtype=np.float32)

            # Clip/temper diversity penalty to prevent extreme forcing
            c = float(getattr(self, "_penalty_clip", 4.0))
            log_ratio = np.clip(log_ratio, -c, c)

            # Adaptive per-cluster theta: stronger for skewed clusters, weaker for already-mixed
            if O_prev is not None:
                kl_k = np.sum(O_prev * (np.log(O_prev + eps) - log_expected), axis=1)  # (K,)
                med_kl = float(np.median(kl_k)) + eps
                ratio = kl_k / med_kl
                ratio = np.clip(ratio, 0.5, 2.0).astype(np.float32, copy=False)
                theta_k = (theta_t * ratio).astype(np.float32, copy=False)  # (K,)
            else:
                theta_k = np.full((K,), theta_t, dtype=np.float32)

            # Permanent sorted-by-batch internal layout: apply penalties directly on contiguous blocks (Change 3)
            ptr = self._batch_ptr
            for b in range(self.n_batches):
                start = int(ptr[b])
                end = int(ptr[b + 1])
                if end <= start:
                    continue
                logR[:, start:end] -= (theta_k * log_ratio[:, b])[:, None]

        # Single softmax (in-place)
        logR -= logR.max(axis=0, keepdims=True)
        np.exp(logR, out=logR)

        # Normalize to get probabilities (in-place)
        denom = logR.sum(axis=0, keepdims=True) + eps
        logR /= denom

        self.R = logR  # already float32

        # R finalized: bump version and invalidate R.T cache
        self._R_version = int(getattr(self, "_R_version", 0)) + 1
        self._R_T_cache = None
        self._R_T_cache_key = None

        def _recompute_S_Rsum():
            # Cache cluster-by-batch weight sums and cluster totals for reuse in objective/correction
            # Permanent sorted-by-batch internal layout: sum contiguous batch blocks directly (Change 3)
            ptr = self._batch_ptr
            S_local = np.add.reduceat(self.R, ptr[:-1], axis=1).astype(np.float32, copy=False)
            self._S = S_local  # (K x B)
            self._R_sum = S_local.sum(axis=1, keepdims=True) + eps  # (K x 1)

        _recompute_S_Rsum()

        # Cache per-cluster KL divergence for correction gating (cluster-aware correction)
        log_expected = getattr(self, "_log_expected", None)
        if log_expected is None:
            expected = self.batch_props[np.newaxis, :].astype(np.float32, copy=False)
            log_expected = np.log(expected + eps).astype(np.float32, copy=False)
        O = self._S / (self._R_sum + eps)  # (K x B)
        kl_k = np.sum(O * (np.log(O + eps) - log_expected), axis=1)  # (K,)
        self._kl_k = kl_k.astype(np.float32, copy=False)

        # Conditional one-step fixed-point refinement for diversity penalty using CURRENT composition
        # Throttle refinement (expensive second softmax pass): early iterations only + stronger guards.
        theta_t = float(getattr(self, "_theta_t", self.theta))
        outer_iter = int(getattr(self, "_outer_iter", -1))
        if outer_iter <= 2 and theta_t > 0.25:
            expected = getattr(self, "_expected", None)
            log_expected = getattr(self, "_log_expected", None)
            if expected is None or log_expected is None:
                expected = self.batch_props[np.newaxis, :].astype(np.float32, copy=False)  # (1 x B)
                log_expected = np.log(expected + eps).astype(np.float32, copy=False)

            O = self._S / (self._R_sum + eps)  # (K x B)

            dev = float(np.max(np.abs(O - expected)))
            dev_threshold = float(getattr(self, "_refine_dev_threshold", 0.10))

            # If mismatch is already small, skip refinement
            if dev > dev_threshold:
                log_ratio = np.log(O + eps) - log_expected  # (K x B)
                c = float(getattr(self, "_penalty_clip", 4.0))
                log_ratio = np.clip(log_ratio, -c, c)

                # If diversity penalty would be tiny anyway, skip refinement
                if float(np.max(np.abs(log_ratio))) >= float(getattr(self, "_refine_logratio_threshold", 0.25)):
                    # Adaptive per-cluster theta based on CURRENT composition
                    kl_k = np.sum(O * (np.log(O + eps) - log_expected), axis=1)  # (K,)
                    med_kl = float(np.median(kl_k)) + eps
                    if med_kl >= float(getattr(self, "_refine_medkl_threshold", 1e-3)):
                        ratio = kl_k / med_kl
                        ratio = np.clip(ratio, 0.5, 2.0).astype(np.float32, copy=False)
                        theta_k = (theta_t * ratio).astype(np.float32, copy=False)  # (K,)

                        # Rebuild logR from distances (already available in `dist`) without recomputing
                        logR[...] = (-dist / sigma_scaled)

                        # Permanent sorted-by-batch internal layout: contiguous blocks (Change 3)
                        ptr = self._batch_ptr
                        for b in range(self.n_batches):
                            start = int(ptr[b])
                            end = int(ptr[b + 1])
                            if end <= start:
                                continue
                            logR[:, start:end] -= (theta_k * log_ratio[:, b])[:, None]

                        # Softmax + normalize
                        logR -= logR.max(axis=0, keepdims=True)
                        np.exp(logR, out=logR)
                        denom = logR.sum(axis=0, keepdims=True) + eps
                        logR /= denom
                        self.R = logR

                        # R changed: bump version and invalidate R.T cache
                        self._R_version = int(getattr(self, "_R_version", 0)) + 1
                        self._R_T_cache = None
                        self._R_T_cache_key = None

                        _recompute_S_Rsum()

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids (allocation-free, versioned cache)."""
        key = (self._Z_version, self._Y_version)
        if self._dist_cache is not None and self._dist_cache_key == key:
            return self._dist_cache

        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 * z @ y
        if self._Z_sq_cache is None or self._Z_sq_cache_key != self._Z_version:
            self._Z_sq_cache = np.sum(self.Z_corr ** 2, axis=1, keepdims=True)  # (n_cells x 1)
            self._Z_sq_cache_key = self._Z_version
        Z_sq = self._Z_sq_cache  # (N x 1)

        if self._Y_sq_cache is None or self._Y_sq_cache_key != self._Y_version:
            self._Y_sq_cache = np.sum(self.Y ** 2, axis=0, keepdims=True)  # (1 x K)
            self._Y_sq_cache_key = self._Y_version
        Y_sq = self._Y_sq_cache  # (1 x K)

        N = self.n_cells if self.n_cells is not None else self.Z_corr.shape[0]
        K = self.n_clusters

        # Allocate/reuse buffers
        if self._cross_buf is None or self._cross_buf.shape != (N, K):
            self._cross_buf = np.empty((N, K), dtype=np.float32)
        if self._dist_buf is None or self._dist_buf.shape != (K, N):
            self._dist_buf = np.empty((K, N), dtype=np.float32)

        # Compute cross = Z @ Y (N x K) into reusable buffer
        try:
            np.matmul(self.Z_corr, self.Y, out=self._cross_buf)
        except TypeError:
            # Fallback for older NumPy where matmul(out=) may be unsupported
            self._cross_buf[...] = self.Z_corr @ self.Y

        # Fill dist_buf in-place:
        # dist = (Z_sq + Y_sq - 2*cross).T
        dist = self._dist_buf
        # dist = -2 * cross.T
        np.multiply(self._cross_buf.T, -2.0, out=dist)

        # Add Z_sq.T (broadcast across K rows)
        dist += Z_sq.T

        # Add Y_sq.T (broadcast across N columns)
        dist += Y_sq.T

        self._dist_cache = dist
        self._dist_cache_key = key
        return dist

    def _correct(self):
        """Apply linear correction to remove batch effects (simultaneous update).

        Optimized to avoid Phi-based dense matmuls:
        - Sort once into contiguous buffers, slice views per batch, scatter once
        - Precompute per-(cluster,batch) weight totals and weighted Z sums via batch-wise GEMMs
        - Solve small ridge systems per cluster
        - Apply batch effects by indexing per-batch (no N x B matmuls)
        - Dampen correction step to reduce over-correction
        - Reuse large buffers to reduce allocations
        - Avoid copying Z_corr; apply update in-place after deltas are computed
        """
        n_cells = self.n_cells if self.n_cells is not None else self.Z_corr.shape[0]
        n_batches = self.n_batches if self.n_batches is not None else (len(self.batch_indices) if self.batch_indices is not None else int(self.batch_id.max() + 1))
        eps = 1e-8

        # Use current Z as base (do not copy); do not mutate until raw_delta computed
        Z_base = self.Z_corr

        lamb = float(self.lamb)
        B = n_batches
        K = self.n_clusters
        F = self.n_features

        # Allocate/reuse buffers
        if getattr(self, "_wb_buf", None) is None or self._wb_buf.shape != (K, B):
            self._wb_buf = np.zeros((K, B), dtype=np.float32)
        else:
            self._wb_buf.fill(0.0)
        wb = self._wb_buf

        # Optional memory reduction vs Zw(K,B,F): keep rhs0(K,F) and rhsb(K,B-1,F)
        if getattr(self, "_rhs0_buf", None) is None or self._rhs0_buf.shape != (K, F):
            self._rhs0_buf = np.zeros((K, F), dtype=np.float32)
        else:
            self._rhs0_buf.fill(0.0)
        rhs0_buf = self._rhs0_buf

        if B > 1:
            if getattr(self, "_rhsb_buf", None) is None or self._rhsb_buf.shape != (K, B - 1, F):
                self._rhsb_buf = np.zeros((K, B - 1, F), dtype=np.float32)
            else:
                self._rhsb_buf.fill(0.0)
            rhsb_buf = self._rhsb_buf
        else:
            rhsb_buf = None

        # NOTE: previous implementation cached a full betas tensor (K,B,F).
        # Vectorized solve below computes only xb(K,B-1,F), so no betas buffer is needed.

        # Cache R_T (N x K) to avoid repeated allocation/copy each correction step
        if self._R_T_cache is None or self._R_T_cache_key != self._R_version:
            self._R_T_cache = np.ascontiguousarray(self.R.T)  # float32 contiguous
            self._R_T_cache_key = self._R_version
        R_T = self._R_T_cache

        # Permanent sorted-by-batch internal layout: Z_base is already batch-contiguous.
        ptr = self._batch_ptr

        # Delta in internal (sorted) space
        if getattr(self, "_delta_sum_buf", None) is None or self._delta_sum_buf.shape != (n_cells, F):
            self._delta_sum_buf = np.zeros((n_cells, F), dtype=np.float32)
        else:
            self._delta_sum_buf.fill(0.0)
        delta = self._delta_sum_buf

        # Precompute wb (K x B) and RHS buffers via batch-wise GEMMs (slice views)
        for b in range(B):
            start = int(ptr[b])
            end = int(ptr[b + 1])
            if end <= start:
                continue
            Wb = R_T[start:end, :]   # view (n_b x K)
            Zb = Z_base[start:end, :]  # view (n_b x F)
            wb[:, b] = Wb.sum(axis=0)
            Zw_kf = Wb.T @ Zb  # (K x F)
            rhs0_buf += Zw_kf
            if b >= 1 and rhsb_buf is not None:
                rhsb_buf[:, b - 1, :] = Zw_kf

        # Vectorized ridge solve across clusters (structured block elimination).
        w_sum = wb.sum(axis=1)  # (K,)
        d0 = lamb + w_sum  # (K,)

        # Cluster-aware correction gating: reduce correction for already well-mixed clusters
        kl_k = getattr(self, "_kl_k", None)
        if kl_k is not None and kl_k.shape[0] == K:
            k50 = int(K // 2)
            med_kl = float(np.partition(kl_k.astype(np.float32, copy=False), k50)[k50]) + eps
            g_k = (kl_k / med_kl).astype(np.float32, copy=False)
            g_k = np.clip(g_k, 0.25, 1.0).astype(np.float32, copy=False)
        else:
            g_k = None

        # Rare-cluster size gating: damp corrections for tiny clusters (Change E)
        k50w = int(K // 2)
        w_med = float(np.partition(w_sum.astype(np.float32, copy=False), k50w)[k50w]) + eps
        size_gate = np.sqrt(w_sum / w_med).astype(np.float32, copy=False)
        min_gate = float(getattr(self, "_min_size_gate", 0.3))
        size_gate = np.clip(size_gate, min_gate, 1.0).astype(np.float32, copy=False)

        if g_k is not None:
            g_k_total = (g_k * size_gate).astype(np.float32, copy=False)
        else:
            g_k_total = size_gate

        if B == 1:
            # Only intercept term; baseline batch => no correction is applied anyway.
            pass
        else:
            db = lamb + wb[:, 1:]  # (K, B-1)
            db = np.maximum(db, eps)
            u = wb[:, 1:]  # (K, B-1)

            u_over_db = u / db  # (K, B-1)
            s = d0 - np.sum(u * u_over_db, axis=1)  # (K,)
            s = np.maximum(s, eps)

            # t = sum_b (u/db) * rhsb over batches dimension -> (K, F)
            t = np.sum(u_over_db[:, :, None] * rhsb_buf, axis=1)  # (K, F)

            x0 = (rhs0_buf - t) / s[:, None]  # (K, F)

            # xb: (K, B-1, F)
            xb = (rhsb_buf - u[:, :, None] * x0[:, None, :]) / db[:, :, None]

            # Apply gating to batch effects (per cluster) (Change E)
            xb *= g_k_total[:, None, None]

            # Apply correction "by batch" (single GEMM per batch) in internal sorted space
            for b in range(1, B):
                start = int(ptr[b])
                end = int(ptr[b + 1])
                if end <= start:
                    continue
                Wb = R_T[start:end, :]      # view (n_b x K)
                Beta_b = xb[:, b - 1, :]    # (K x F)
                delta[start:end, :] = Wb @ Beta_b

        raw_delta = delta

        # Correction magnitude clipping (robust per-iteration guardrail) - per cell
        dn = np.linalg.norm(raw_delta, axis=1)

        # Use partition instead of percentile (median and ~97.5%)
        k50 = int(n_cells // 2)
        k975 = int(0.975 * float(n_cells - 1))
        dn_part = np.partition(dn, (k50, k975))
        self._last_median_delta = float(dn_part[k50])
        cap = float(dn_part[k975]) * 1.5
        if cap > 0:
            scale = np.minimum(1.0, cap / (dn + eps)).astype(np.float32, copy=False)
            raw_delta *= scale[:, None]

        # Additional per-feature/PC clipping to prevent a few dimensions from being over-shifted.
        # Update caps once per outer iteration using a subsample for speed.
        outer_iter = int(getattr(self, "_outer_iter", -1))
        if self._feat_cap is None or self._feat_clip_outer_iter != outer_iter:
            n_sub = int(min(1024, n_cells))
            if n_sub > 0:
                if self._feat_cap_sub_idx_outer_iter != outer_iter or self._feat_cap_sub_idx is None or self._feat_cap_sub_idx.size != n_sub:
                    self._feat_cap_sub_idx = self._rng.choice(n_cells, size=n_sub, replace=False)
                    self._feat_cap_sub_idx_outer_iter = outer_iter
                sub_idx = self._feat_cap_sub_idx
                p = np.percentile(np.abs(raw_delta[sub_idx, :]), 97.5, axis=0).astype(np.float32, copy=False)
                self._feat_cap = (1.5 * p).astype(np.float32, copy=False)
            else:
                self._feat_cap = np.zeros((F,), dtype=np.float32)
            self._feat_clip_outer_iter = outer_iter

        feat_cap = self._feat_cap
        if feat_cap is not None:
            np.clip(raw_delta, -feat_cap[None, :], feat_cap[None, :], out=raw_delta)

        alpha = float(getattr(self, "_alpha", 1.0))
        # In-place update (no Z_base copy)
        self.Z_corr -= (alpha * raw_delta).astype(np.float32, copy=False)

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
        if self._S is not None:
            S = self._S
        else:
            S = np.zeros((self.n_clusters, self.n_batches), dtype=np.float32)
            if self.batch_indices is None:
                self.batch_indices = [np.where(self.batch_id == b)[0] for b in range(self.n_batches)]
            for b in range(self.n_batches):
                idx = self.batch_indices[b]
                if idx.size == 0:
                    continue
                S[:, b] = self.R[:, idx].sum(axis=1)
        O = S / R_sum
        log_expected = getattr(self, "_log_expected", None)
        if log_expected is None:
            expected = self.batch_props[np.newaxis, :].astype(np.float32, copy=False)
            log_expected = np.log(expected + 1e-8).astype(np.float32, copy=False)
        theta_t = float(getattr(self, "_theta_t", self.theta))
        diversity_obj = theta_t * np.sum(
            O * (np.log(O + 1e-8) - log_expected)
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
