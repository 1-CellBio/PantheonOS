import warnings
import pandas as pd
import numpy as np
import scipy
import types
import sys
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from umap.umap_ import fuzzy_simplicial_set
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree

# Optional imports
try:
	from annoy import AnnoyIndex
except ImportError:
	AnnoyIndex = None
try:
	import pynndescent
except ImportError:
	pynndescent = None
try:
	from scanpy import logging as logg
except ImportError:
	pass
try:
	import faiss
except ImportError:
	pass

def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
	'''
	Copied out of scanpy.neighbors

	Vectorized for performance.
	'''
	# vectorized COO construction
	rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
	cols = knn_indices.reshape(-1).astype(np.int64, copy=False)
	vals = knn_dists.reshape(-1).astype(np.float64, copy=False)

	# filter out missing neighbors
	keep = cols != -1
	if not np.all(keep):
		rows = rows[keep]
		cols = cols[keep]
		vals = vals[keep]

	# scanpy-compatible: self edges get 0 distance (and will be eliminated below)
	self_mask = cols == rows
	if np.any(self_mask):
		vals = vals.copy()
		vals[self_mask] = 0.0

	result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
	result.eliminate_zeros()
	return result.tocsr()

def compute_connectivities_umap(knn_indices, knn_dists,
		n_obs, n_neighbors, set_op_mix_ratio=1.0,
		local_connectivity=1.0):
	'''
	Compute distances and connectivities from (knn_indices, knn_dists).

	Historically this was copied out of scanpy.neighbors and delegated to
	UMAP's fuzzy_simplicial_set. For BBKNN, edge selection is the core
	correction step and UMAP's fuzzy set can amplify any remaining false
	positive cross-batch edges.

	This function now supports an internal "self-tuning kernel" mode
	(Zelnik-Manor & Perona) with optional cross-batch damping, controlled via
	module-level state set by bbknn().

	Public signature remains unchanged for compatibility.
	'''
	# always return distances in scanpy-compatible format
	distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

	# module-level state injected by bbknn(); keeps public API stable
	global _BBKNN_CONNECTIVITY_MODE, _BBKNN_BATCH_LIST, _BBKNN_CROSS_DAMPING, _BBKNN_BATCH_CODES, _BBKNN_N_BATCHES
	mode = globals().get('_BBKNN_CONNECTIVITY_MODE', 'kernel')
	batch_list = globals().get('_BBKNN_BATCH_LIST', None)
	use_damping = bool(globals().get('_BBKNN_CROSS_DAMPING', True))

	if mode == 'umap':
		X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
		connectivities = fuzzy_simplicial_set(
			X, n_neighbors, None, None,
			knn_indices=knn_indices, knn_dists=knn_dists,
			set_op_mix_ratio=set_op_mix_ratio,
			local_connectivity=local_connectivity,
		)
		if isinstance(connectivities, tuple):
			# In umap-learn 0.4, this returns (result, sigmas, rhos)
			connectivities = connectivities[0]
		return distances, connectivities.tocsr()

	# -------------------------
	# Self-tuning kernel graph:
	# w_ij = exp( - d_ij^2 / (sigma_i * sigma_j + eps) )
	# sigma_i = d(i, k_sigma) with k_sigma chosen from local_connectivity / n_neighbors.
	# plus optional cross-batch damping using a single-pass attenuation.
	# -------------------------
	d = np.asarray(knn_dists, dtype=np.float64)
	# choose k_sigma: use ~10th neighbour when available, otherwise fall back.
	k_sigma = int(min(max(10, int(np.ceil(local_connectivity))), n_neighbors) - 1)
	k_sigma = max(0, min(k_sigma, n_neighbors - 1))
	sigma = d[:, k_sigma].copy()
	# robustify sigma (avoid zeros for identical points)
	sigma[sigma <= 0] = np.median(d[d > 0]) if np.any(d > 0) else 1.0

	# build edge list for kernel weights
	rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
	cols = knn_indices.reshape(-1).astype(np.int64, copy=False)
	d_flat = d.reshape(-1)

	sigma_i = np.repeat(sigma, n_neighbors)
	sigma_j = sigma[cols]
	den = (sigma_i * sigma_j) + 1e-12

	# exponent clipping for numerical stability (avoid underflow -> all-zero rows)
	exponent = -((d_flat * d_flat) / den)
	exponent = np.maximum(exponent, -60.0)
	w = np.exp(exponent)

	# ------------------------------------------------------------
	# Light-weight cross-batch attenuation (single pass; no IPF).
	# Optionally skipped to avoid double-penalisation when cross-batch selection already
	# enforced conservative links.
	# ------------------------------------------------------------
	global _BBKNN_DAMP_AFTER_SELECTION
	damp_after_selection = bool(globals().get('_BBKNN_DAMP_AFTER_SELECTION', True))

	if batch_list is not None and use_damping and damp_after_selection:
		batch_arr = np.asarray(batch_list)

		batch_codes = globals().get('_BBKNN_BATCH_CODES', None)
		B = int(globals().get('_BBKNN_N_BATCHES', 0) or 0)

		# fall back if codes were not provided
		if batch_codes is None or B <= 0:
			_, batch_codes = np.unique(batch_arr, return_inverse=True)
			batch_codes = batch_codes.astype(np.int64, copy=False)
			B = int(batch_codes.max() + 1) if batch_codes.size else 0

		# per-edge batch id (of the target node)
		edge_batch_ind = batch_codes[cols]

		# mixing allowance m_i in [0,1]: high when cross-batch is comparable to within-batch
		Dm = d_flat.reshape((n_obs, n_neighbors))
		Cm = cols.reshape((n_obs, n_neighbors))
		within = (batch_codes[Cm] == batch_codes[:, None])

		D_within = np.where(within, Dm, np.nan)
		D_cross = np.where(~within, Dm, np.nan)

		d_within = np.nanmedian(D_within, axis=1)
		d_cross = np.nanmedian(D_cross, axis=1)

		global_med = np.median(d_flat) if d_flat.size else 1.0
		d_within = np.where(np.isfinite(d_within), d_within, global_med)
		d_cross = np.where(np.isfinite(d_cross), d_cross, global_med)

		eps = 1e-12
		gap = (d_cross - d_within) / (d_within + eps)
		# convert to allowance: gap<=0 -> ~1, large gap -> ~0
		m_i = 1.0 / (1.0 + np.maximum(gap, 0.0))
		m_i = np.clip(m_i, 0.0, 1.0)

		# attenuation factor for cross-batch edges
		lam = 0.35
		cross_edge = (edge_batch_ind != np.repeat(batch_codes, n_neighbors))
		att = 1.0 - (lam * (1.0 - np.repeat(m_i, n_neighbors)) * cross_edge.astype(np.float64))
		att = np.clip(att, 0.0, 1.0)
		w *= att

	connectivities = coo_matrix((w, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
	connectivities.eliminate_zeros()
	# symmetrise with max to get an undirected graph
	connectivities = connectivities.maximum(connectivities.T).tocsr()
	connectivities.eliminate_zeros()

	return distances, connectivities

def create_tree(data, params):
	'''
	Create a faiss/cKDTree/KDTree/annoy/pynndescent index for nearest neighbour lookup. 
	All undescribed input as in ``bbknn.bbknn()``. Returns the resulting index.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to index.
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckd = AnnoyIndex(data.shape[1],metric=params['metric'])
		for i in np.arange(data.shape[0]):
			ckd.add_item(i,data[i,:])
		ckd.build(params['annoy_n_trees'])
	elif params['computation'] == 'pynndescent':
		ckd = pynndescent.NNDescent(data, metric=params['metric'], n_jobs=-1,
									n_neighbors=params['pynndescent_n_neighbors'], 
									random_state=params['pynndescent_random_state'])
		ckd.prepare()
	elif params['computation'] == 'faiss':
		try: # uses GPU if faiss_gpu available.
			res = faiss.StandardGpuResources()
			ckd_ = faiss.IndexFlatL2(data.shape[1])
			ckd = faiss.index_cpu_to_gpu(res, 0, ckd_)
		except:
			ckd = faiss.IndexFlatL2(data.shape[1])
		ckd.add(data)
	elif params['computation'] == 'cKDTree':
		ckd = cKDTree(data)
	elif params['computation'] == 'KDTree':
		ckd = KDTree(data,metric=params['metric'])
	return ckd

def query_tree(data, ckd, params):
	'''
	Query the faiss/cKDTree/KDTree/annoy index with PCA coordinates from a batch. All undescribed input
	as in ``bbknn.bbknn()``. Returns a tuple of distances and indices of neighbours for each cell
	in the batch.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to query.
	ckd : faiss/cKDTree/KDTree/annoy/pynndescent index
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckdo_ind = []
		ckdo_dist = []
		for i in np.arange(data.shape[0]):
			holder = ckd.get_nns_by_vector(data[i,:],params['neighbors_within_batch'],include_distances=True)
			ckdo_ind.append(holder[0])
			ckdo_dist.append(holder[1])
		ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
	elif params['computation'] == 'pynndescent':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
		ckdout = (ckdout[1], ckdout[0])
	elif params['computation'] == 'faiss':
		D, I = ckd.search(data, params['neighbors_within_batch'])
		#sometimes this turns up marginally negative values, just set those to zero
		D[D<0] = 0
		#the distance returned by faiss needs to be square rooted to be actual euclidean
		ckdout = (np.sqrt(D), I)
	elif params['computation'] == 'cKDTree':
		ckdout = ckd.query(x=data, k=params['neighbors_within_batch'], workers=-1)
	elif params['computation'] == 'KDTree':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
	return ckdout

def get_graph(pca, batch_list, params):
	'''
	Identify the KNN structure to be used in graph construction.

	Updated implementation:

	1) Within-batch backbone (biological anchor):
	   For each cell, take k_in = neighbors_within_batch neighbours from its *own* batch.

	2) Cross-batch selection via pooled candidates + quota-constrained assignment (Sinkhorn/OT-style):
	   For each source batch, query each other batch for k' candidates. Pool candidates per cell,
	   then select k_out cross-batch neighbours with approximately balanced batch representation
	   using a small entropy-regularized assignment over candidates.

	   Quotas are adaptive per cell to avoid forced mixing when cross-batch candidates are much
	   farther than within-batch neighbours.

	Returns (knn_distances, knn_indices) with a fixed neighbour count per cell.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	n_batches = len(batches)

	# in case we're gonna be faissing, turn the data to float32
	if params['computation'] == 'faiss':
		pca = pca.astype('float32')

	n_cells = pca.shape[0]
	k_in = int(params['neighbors_within_batch'])
	k_total = int(params['neighbors_within_batch'] * n_batches)
	k_out = int(max(k_total - k_in, 0))

	# cross-batch candidate pool size (k')
	k_cross_candidates = int(max(3 * k_in, k_in))

	# precompute per-batch indices for mapping between local and global ids
	batch_arr = np.asarray(batch_list)
	batch_to_inds = {}
	for b in batches:
		batch_to_inds[b] = np.where(batch_arr == b)[0]

	# cache per-batch matrices + trees once and reuse everywhere
	batch_X = {}
	batch_tree = {}
	for b in batches:
		inds = batch_to_inds[b]
		Xb = pca[inds, :params['n_pcs']]
		batch_X[b] = Xb
		batch_tree[b] = create_tree(data=Xb, params=params)

	# within-batch backbone: query within each batch (k_in + 1), then drop self-neighbour robustly
	within_indices = np.zeros((n_cells, k_in), dtype=np.int64)
	within_dists = np.zeros((n_cells, k_in), dtype=np.float64)

	params_in = params.copy()
	params_in['neighbors_within_batch'] = k_in + 1

	for b in batches:
		inds = batch_to_inds[b]
		Xb = batch_X[b]
		ckd = batch_tree[b]
		d_loc, i_loc = query_tree(data=Xb, ckd=ckd, params=params_in)
		d_loc = np.asarray(d_loc, dtype=np.float64)
		i_loc = np.asarray(i_loc, dtype=np.int64)

		# map local->global
		g_inds = inds[i_loc]

		# drop self hits per row and keep k_in smallest distances using argpartition
		self_mask = (g_inds == inds[:, None])
		# set self to +inf distance so it is never selected
		if np.any(self_mask):
			d_loc = d_loc.copy()
			d_loc[self_mask] = np.inf

		# select k_in candidates without fully sorting
		sel = np.argpartition(d_loc, kth=(k_in - 1), axis=1)[:, :k_in]
		row = np.arange(d_loc.shape[0])[:, None]
		# stable ordering within the selected set
		sel_sorted = sel[row, np.argsort(d_loc[row, sel], axis=1)]

		within_indices[inds, :] = g_inds[row, sel_sorted]
		within_dists[inds, :] = d_loc[row, sel_sorted]

	# per-cell final neighbour lists
	knn_indices = np.zeros((n_cells, k_total), dtype=np.int64)
	knn_distances = np.zeros((n_cells, k_total), dtype=np.float64)

	# initialise with within-batch backbone
	knn_indices[:, :k_in] = within_indices
	knn_distances[:, :k_in] = within_dists

	if k_out <= 0 or n_batches <= 1:
		return knn_distances, knn_indices

	# per-other-batch soft quota baseline
	n_other = max(1, (n_batches - 1))
	base_quota = k_out / float(n_other)

	# --- adaptive mixing allowance m_i in [0,1] based on within vs cross distance gap ---
	with_d = np.asarray(within_dists, dtype=np.float64)
	d_within_med = np.median(with_d, axis=1)
	# robustify (avoid zero / nan)
	global_with_med = np.median(with_d[np.isfinite(with_d) & (with_d > 0)]) if np.any(np.isfinite(with_d) & (with_d > 0)) else 1.0
	d_within_med = np.where(np.isfinite(d_within_med) & (d_within_med > 0), d_within_med, global_with_med)

	# helper for per-cell selection from pooled candidates using quota-constrained scaling
	def _select_cross_edges_for_batch(src_global_inds, pooled_j, pooled_d, pooled_b, m_i_src, k_out, B_other):
		'''
		src_global_inds: (n_src,) global indices
		pooled_j: (n_src, n_cand) global neighbour ids
		pooled_d: (n_src, n_cand) distances
		pooled_b: (n_src, n_cand) target-batch code in [0..B_other-1]
		m_i_src: (n_src,) mixing allowance in [0,1]
		Returns: cross_idx (n_src,k_out), cross_dist (n_src,k_out)
		'''
		n_src, n_cand = pooled_j.shape
		eps = 1e-12

		# per-row scale for RBF affinity (sigma_i)
		sig = np.quantile(pooled_d, 0.5, axis=1)
		global_sig = np.median(sig[np.isfinite(sig) & (sig > 0)]) if np.any(np.isfinite(sig) & (sig > 0)) else 1.0
		sig = np.where(np.isfinite(sig) & (sig > 0), sig, global_sig)

		# base affinity from distance
		A = np.exp(-((pooled_d * pooled_d) / ((sig[:, None] * sig[:, None]) + eps)))
		A = np.maximum(A, 1e-300)

		# per-row quotas over other batches:
		Q = np.zeros((n_src, B_other), dtype=np.float64)
		if B_other <= 1:
			Q[:, 0] = float(k_out)
		else:
			# candidates are packed contiguously by target batch:
			# batch b occupies [b*k_cross_candidates : (b+1)*k_cross_candidates]
			k_block = int(n_cand // B_other) if B_other else n_cand
			best_d = np.full((n_src, B_other), np.inf, dtype=np.float64)
			for b in range(B_other):
				start = b * k_block
				end = start + k_block
				best_d[:, b] = np.min(pooled_d[:, start:end], axis=1)
			best_b = np.argmin(best_d, axis=1)

			unif = (m_i_src * (float(k_out) / float(B_other)))
			Q[:] = unif[:, None]
			Q[np.arange(n_src), best_b] += (1.0 - m_i_src) * float(k_out)

		# single-pass per-batch reweighting (soft quota enforcement) to avoid iterative sinkhorn
		if B_other > 1:
			k_block = int(n_cand // B_other)
			for b in range(B_other):
				start = b * k_block
				end = start + k_block
				Sb = np.sum(A[:, start:end], axis=1)
				lam = Q[:, b] / (Sb + eps)
				lam = np.clip(lam, 0.25, 4.0)
				A[:, start:end] *= lam[:, None]

		# now select top-k_out by scaled affinity
		if n_cand <= k_out:
			choose = np.argsort(-A, axis=1)
			choose = np.pad(choose, ((0, 0), (0, max(0, k_out - n_cand))), mode='edge')[:, :k_out]
		else:
			choose = np.argpartition(A, -k_out, axis=1)[:, -k_out:]
			row = np.arange(n_src)[:, None]
			order = np.argsort(-A[row, choose], axis=1)
			choose = choose[row, order]

		cross_idx = pooled_j[np.arange(n_src)[:, None], choose].astype(np.int64, copy=False)
		cross_dist = pooled_d[np.arange(n_src)[:, None], choose].astype(np.float64, copy=False)
		return cross_idx, cross_dist

	# pooled candidate queries, per source batch
	params_cross = params.copy()
	params_cross['neighbors_within_batch'] = k_cross_candidates

	# precompute global batch codes for fast comparisons
	_, batch_codes = np.unique(batch_arr, return_inverse=True)
	batch_codes = batch_codes.astype(np.int64, copy=False)

	for b_src in batches:
		src_inds = batch_to_inds[b_src]
		n_src = src_inds.shape[0]
		X_src = batch_X[b_src]

		# gather candidates from all other batches
		other_batches = [b for b in batches if b != b_src]
		B_other = len(other_batches)

		# each target contributes k_cross_candidates; pool into (n_src, n_cand)
		n_cand = int(B_other * k_cross_candidates)
		pooled_j = np.empty((n_src, n_cand), dtype=np.int64)
		pooled_d = np.empty((n_src, n_cand), dtype=np.float64)
		pooled_b = np.empty((n_src, n_cand), dtype=np.int64)

		offset = 0
		for b_idx, b_tgt in enumerate(other_batches):
			d_st, i_st = query_tree(data=X_src, ckd=batch_tree[b_tgt], params=params_cross)
			d_st = np.asarray(d_st, dtype=np.float64)
			i_st = np.asarray(i_st, dtype=np.int64)

			j_global = batch_to_inds[b_tgt][i_st]
			pooled_j[:, offset:offset + k_cross_candidates] = j_global
			pooled_d[:, offset:offset + k_cross_candidates] = d_st
			pooled_b[:, offset:offset + k_cross_candidates] = b_idx
			offset += k_cross_candidates

		# prevent accidental self edges from approximate methods
		self_cand = (pooled_j == src_inds[:, None])
		if np.any(self_cand):
			pooled_d = pooled_d.copy()
			pooled_d[self_cand] = np.inf

		# compute m_i for these source cells using pooled distances
		d_cross_med = np.median(pooled_d, axis=1)
		eps = 1e-12
		gap = (d_cross_med - d_within_med[src_inds]) / (d_within_med[src_inds] + eps)
		m_i_src = 1.0 / (1.0 + np.maximum(gap, 0.0))
		m_i_src = np.clip(m_i_src, 0.0, 1.0)

		# select cross edges by quota-constrained scaling
		cross_idx, cross_dist = _select_cross_edges_for_batch(
			src_global_inds=src_inds,
			pooled_j=pooled_j,
			pooled_d=pooled_d,
			pooled_b=pooled_b,
			m_i_src=m_i_src,
			k_out=k_out,
			B_other=B_other,
		)

		# store into final knn arrays after within backbone
		knn_indices[src_inds, k_in:(k_in + k_out)] = cross_idx
		knn_distances[src_inds, k_in:(k_in + k_out)] = cross_dist

	# final cleanup (vectorized): ensure no self edges
	row_ids = np.arange(n_cells, dtype=np.int64)[:, None]
	self_mask = (knn_indices == row_ids)
	if np.any(self_mask):
		# replace any accidental self entries with furthest within-batch neighbour
		fallback_idx = knn_indices[:, k_in - 1:k_in]
		fallback_dist = knn_distances[:, k_in - 1:k_in]
		knn_indices = knn_indices.copy()
		knn_distances = knn_distances.copy()
		knn_indices[self_mask] = np.broadcast_to(fallback_idx, knn_indices.shape)[self_mask]
		knn_distances[self_mask] = np.broadcast_to(fallback_dist, knn_distances.shape)[self_mask]

	return knn_distances, knn_indices

def print_warning(message, scanpy_logging):
	'''
	Print a specified warning to the screen, using either warnings or scanpy.
	
	Input
	-----
	message : ``str``
		The warning message to print
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	if scanpy_logging:
		logg.warning(message)
	else:
		warnings.warn(message)

def legacy_computation_selection(params, scanpy_logging):
	'''
	Do pre-1.6.0 computation algorithm selection based on possible legacy arguments. Looks 
	at the following (default None) parameters: approx, use_annoy, use_faiss
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#if these are not None then they were set at the time of the call
	if any([params[i] is not None for i in ['approx','use_annoy','use_faiss']]):
		#encourage upgrading the call
		print_warning('consider updating your call to make use of `computation`', scanpy_logging)
		#fill in any missing defaults
		if params['approx'] is None:
			params['approx'] = True
		if params['use_annoy'] is None:
			params['use_annoy'] = True
		if params['use_faiss'] is None:
			params['use_faiss'] = True
		#now that we have all the old defaults restored, use them to pick a computation
		if params['approx']:
			#pick between these two packages based on another param
			if params['use_annoy']:
				params['computation'] = 'annoy'
			else:
				params['computation'] = 'pynndescent'
		else:
			#if the metric is euclidean, then pick between these two packages
			if params['metric'] == 'euclidean':
				if params['use_faiss'] and 'faiss' in sys.modules:
					params['computation'] = 'faiss'
				else:
					params['computation'] = 'cKDTree'
			else:
				params['computation'] = 'KDTree'
	return params

def check_knn_metric(params, counts, scanpy_logging):
	'''
	Checks if the provided metric can be used with the implied KNN algorithm. Returns parameters
	with the metric validated.
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``
	counts : ``np.array``
		The number of cells in each batch
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#take note if we end up going back to Euclidean
	swapped = False
	if params['computation'] == 'annoy':
		if params['metric'] not in ['angular', 'euclidean', 'manhattan', 'hamming']:
			swapped = True
	elif params['computation'] == 'pynndescent':
		if np.min(counts) < 11:
			raise ValueError("Not all batches have at least 11 cells in them - required by pynndescent.")
		#metric needs to be a function or in the named list
		if not (params['metric'] in pynndescent.distances.named_distances or
				isinstance(params['metric'], types.FunctionType)):
			swapped = True
	elif params['computation'] == 'faiss':
		if params['metric'] != 'euclidean':
			swapped = True
	elif params['computation'] == 'cKDTree':
		if params['metric'] != 'euclidean':
			swapped = True
	elif params['computation'] == 'KDTree':
		if not (isinstance(params['metric'], DistanceMetric) or 
				params['metric'] in KDTree.valid_metrics):
			swapped = True
			#and now for a next level swap - this can be done more efficiently via cKDTree
			params['computation'] = 'cKDTree'
	else:
		raise ValueError("Incorrect value of `computation`.")
	if swapped:
		#go back to euclidean
		params['metric'] = 'euclidean'
		#need to let the user know we swapped the metric
		print_warning('unrecognised metric for type of neighbor calculation, switching to euclidean', scanpy_logging)
	return params

def trimming(cnts, trim):
	'''
	Trims the graph using diffusion-based denoising + sparsification.

	Exploration-phase strategy:
	1) Start from connectivities W (sparse).
	2) Optionally build a short diffusion affinity using a few power iterations of a random walk:
	   P = D^{-1} W, then A ≈ sum_{t=1..k} alpha^(t-1) P^t.
	3) Sparsify by keeping top-`trim` affinities per row, then symmetrise.

	Input
	-----
	cnts : ``CSR``
		Sparse matrix of processed connectivities to trim.
	trim : ``int``
		Maximum number of edges to keep per row (before final symmetrisation).
	'''
	if trim is None or trim <= 0:
		return cnts

	W = cnts.tocsr()
	W.eliminate_zeros()
	n = W.shape[0]

	row_nnz = np.diff(W.indptr)
	if row_nnz.size and (np.mean(row_nnz) <= trim):
		return W

	# skip diffusion if graph isn't much denser than trim (reduce oversmoothing + cost)
	use_diffusion = True
	if row_nnz.size:
		med_deg = np.median(row_nnz)
		if med_deg <= (1.5 * trim):
			use_diffusion = False

	if use_diffusion:
		# row-normalize to get transition matrix P
		rs = np.asarray(W.sum(axis=1)).ravel()
		rs[rs == 0] = 1.0
		inv_rs = 1.0 / rs
		P = scipy.sparse.diags(inv_rs).dot(W).tocsr()
		P.eliminate_zeros()

		# short diffusion (PPR-like truncated series)
		alpha = 0.5
		A = P.copy().tocsr()
		Pt = P.dot(P).tocsr()
		Pt.eliminate_zeros()
		A = (A + alpha * Pt).tocsr()
		A.eliminate_zeros()
	else:
		A = W

	# CSR top-k pruning with reduced Python overhead
	A = A.tocsr()
	A.eliminate_zeros()
	indptr = A.indptr
	indices = A.indices
	data = A.data

	nnz = np.diff(indptr)
	keep_counts = np.minimum(nnz, trim).astype(np.int64, copy=False)
	out_nnz = int(keep_counts.sum())

	out_indptr = np.empty(n + 1, dtype=np.int64)
	out_indptr[0] = 0
	np.cumsum(keep_counts, out=out_indptr[1:])

	out_indices = np.empty(out_nnz, dtype=np.int64)
	out_data = np.empty(out_nnz, dtype=np.float64)

	# copy rows with nnz <= trim directly; prune only rows needing it
	rows_need = np.flatnonzero(nnz > trim)
	rows_ok = np.flatnonzero((nnz > 0) & (nnz <= trim))

	for i in rows_ok:
		start, end = indptr[i], indptr[i + 1]
		o_start, o_end = out_indptr[i], out_indptr[i + 1]
		out_indices[o_start:o_end] = indices[start:end]
		out_data[o_start:o_end] = data[start:end]

	for i in rows_need:
		start, end = indptr[i], indptr[i + 1]
		r_idx = indices[start:end]
		r_dat = data[start:end]
		k = int(trim)

		choose = np.argpartition(r_dat, -k)[-k:]
		order = np.argsort(r_dat[choose])[::-1]
		choose = choose[order]

		o_start, o_end = out_indptr[i], out_indptr[i + 1]
		out_indices[o_start:o_end] = r_idx[choose].astype(np.int64, copy=False)
		out_data[o_start:o_end] = r_dat[choose].astype(np.float64, copy=False)

	trimmed = scipy.sparse.csr_matrix((out_data, out_indices, out_indptr), shape=W.shape)
	trimmed.eliminate_zeros()

	# symmetrise: keep the maximum affinity between i<->j
	trimmed = trimmed.maximum(trimmed.T).tocsr()
	trimmed.eliminate_zeros()
	return trimmed

def bbknn(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None,
		  computation='annoy', annoy_n_trees=10, pynndescent_n_neighbors=30, 
		  pynndescent_random_state=0, metric='euclidean', set_op_mix_ratio=1, 
		  local_connectivity=1, approx=None, use_annoy=None, use_faiss=None,
		  scanpy_logging=False):
	'''
	Scanpy-independent BBKNN variant that runs on a PCA matrix and list of per-cell batch assignments instead of
	an AnnData object. Non-data-entry arguments behave the same way as ``bbknn.bbknn()``.
	Returns a ``(distances, connectivities, parameters)`` tuple, like what would have been stored in the AnnData object.
	The connectivities are the actual neighbourhood graph.

	Input
	-----
	pca : ``numpy.array``
		PCA (or other dimensionality reduction) coordinates for each cell, with cells as rows.
	batch_list : ``numpy.array`` or ``list``
		A list of batch assignments for each cell.
	scanpy_logging : ``bool``, optional (default: ``False``)
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#catch all arguments for easy passing to subsequent functions
	params = locals()
	del params['pca']
	del params['batch_list']
	#more basic sanity checks/processing
	#do we have the same number of cells in pca and batch_list?
	if pca.shape[0] != len(batch_list):
		raise ValueError("Different cell counts indicated by `pca.shape[0]` and `len(batch_list)`.")
	#make sure n_pcs is not larger than the actual dimensions of the data
	#as that can introduce some uninformative knock-on errors in UMAP
	params['n_pcs'] = np.min([params['n_pcs'], pca.shape[1]])
	#convert batch_list to np.array of strings for ease of mask making later
	batch_list = np.asarray([str(i) for i in batch_list])
	# expose batch_list to connectivity construction without changing public APIs
	global _BBKNN_BATCH_LIST, _BBKNN_CONNECTIVITY_MODE, _BBKNN_CROSS_DAMPING, _BBKNN_BATCH_CODES, _BBKNN_N_BATCHES, _BBKNN_DAMP_AFTER_SELECTION
	_BBKNN_BATCH_LIST = batch_list
	# precompute categorical batch codes for fast batch mapping in connectivity construction
	_, batch_codes = np.unique(batch_list, return_inverse=True)
	_BBKNN_BATCH_CODES = batch_codes.astype(np.int64, copy=False)
	_BBKNN_N_BATCHES = int(_BBKNN_BATCH_CODES.max() + 1) if _BBKNN_BATCH_CODES.size else 0
	# use self-tuning kernel connectivities by default (more conservative than UMAP fuzzy set)
	_BBKNN_CONNECTIVITY_MODE = 'kernel'
	# enable adaptive cross-batch damping (may be skipped after selection; see flag below)
	_BBKNN_CROSS_DAMPING = True
	# avoid double-penalisation: get_graph() already performs conservative cross-batch selection
	_BBKNN_DAMP_AFTER_SELECTION = False
	#assert that all batches have at least neighbors_within_batch cells in there
	unique, counts = np.unique(batch_list, return_counts=True)
	if np.min(counts) < params['neighbors_within_batch']:
		raise ValueError("Not all batches have at least `neighbors_within_batch` cells in them.")
	#if any of the old legacy computation selection arguments are detected
	#use them to select the knn computation algorithm
	params = legacy_computation_selection(params, scanpy_logging)
	#sanity check the metric
	params = check_knn_metric(params, counts, scanpy_logging)
	#obtain the batch balanced KNN graph
	knn_distances, knn_indices = get_graph(pca=pca,batch_list=batch_list,params=params)
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx]
	#this part of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0],
											 knn_indices.shape[1], set_op_mix_ratio=set_op_mix_ratio,
											 local_connectivity=local_connectivity)
	#trimming. compute default range if absent
	#both this and the parameter dictionary need a neighbour total for each cell
	#easiest retrieved from the shape of the knn variables
	if params['trim'] is None:
		params['trim'] = 10 * knn_distances.shape[1]
	#skip trimming if set to 0, otherwise trim
	if params['trim'] > 0:
		cnts = trimming(cnts=cnts,trim=params['trim'])
	#create a collated parameters dictionary, formatted like scanpy's neighbours one
	p_dict = {'n_neighbors': knn_distances.shape[1], 'method': 'umap', 
			  'metric': params['metric'], 'n_pcs': params['n_pcs'], 
			  'bbknn': {'trim': params['trim'], 'computation': params['computation']}}
	return (dist, cnts, p_dict)
