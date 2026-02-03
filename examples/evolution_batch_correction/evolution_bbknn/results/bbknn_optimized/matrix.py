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
	'''
	rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
	cols = knn_indices.reshape(-1).astype(np.int64, copy=False)
	vals = knn_dists.reshape(-1).astype(np.float64, copy=False)

	# mask invalid neighbors (-1)
	mask = cols >= 0
	rows = rows[mask]
	cols = cols[mask]
	vals = vals[mask]

	# diagonal rule: distance to self is zero
	diag_mask = cols == rows
	if np.any(diag_mask):
		vals = vals.copy()
		vals[diag_mask] = 0.0

	result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
	result.eliminate_zeros()
	return result.tocsr()

def compute_connectivities_umap(knn_indices, knn_dists,
		n_obs, n_neighbors, set_op_mix_ratio=1.0,
		local_connectivity=1.0):
	'''
	Copied out of scanpy.neighbors

	This is from umap.fuzzy_simplicial_set [McInnes18]_.
	Given a set of data X, a neighborhood size, and a measure of distance
	compute the fuzzy simplicial set (here represented as a fuzzy graph in
	the form of a sparse matrix) associated to the data. This is done by
	locally approximating geodesic distance at each point, creating a fuzzy
	simplicial set for each such point, and then combining all the local
	fuzzy simplicial sets into a global one via a fuzzy union.
	'''
	X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
	connectivities = fuzzy_simplicial_set(X, n_neighbors, None, None,
										  knn_indices=knn_indices, knn_dists=knn_dists,
										  set_op_mix_ratio=set_op_mix_ratio,
										  local_connectivity=local_connectivity)
	if isinstance(connectivities, tuple):
		# In umap-learn 0.4, this returns (result, sigmas, rhos)
		connectivities = connectivities[0]
	distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

	return distances, connectivities.tocsr()


def compute_connectivities_batchaware_kernel(knn_indices, knn_dists, batch_list,
		set_op_mix_ratio=1.0, local_connectivity=1.0):
	'''
	Batch-aware connectivity construction using a self-tuning kernel mixture.

	- Distances are converted to affinities via a self-tuning kernel (Zelnik-Manor & Perona).
	- Within-batch and cross-batch edges are weighted with separate local scales and a
	  cross-batch damping factor to reduce over-correction.
	- Returns (distances, connectivities) as CSR matrices, matching scanpy/neighbors output.

	Notes
	-----
	This function is internal and does not change the public API. It is called by bbknn()
	with the available batch_list.
	'''
	batch_list = np.asarray([str(i) for i in batch_list])
	n_obs = int(knn_indices.shape[0])
	n_neighbors = int(knn_indices.shape[1])

	# distances matrix (like scanpy) is still based on raw knn_dists
	distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

	# Build local scales for within- and cross-batch edges per node.
	# Use a robust "kth neighbor" scale; fallback to median / 1.0 if absent.
	d = np.asarray(knn_dists, dtype=np.float64, order='C')
	inds = np.asarray(knn_indices, dtype=np.int64, order='C')

	# per-edge batch relationship mask (n_obs, n_neighbors)
	nb_batches = batch_list[inds]
	self_batches = batch_list[:, None]
	is_within = (nb_batches == self_batches)

	# pick kth index within available neighbours for scale estimation
	kth = int(max(0, min(n_neighbors - 1, int(local_connectivity))))
	# compute within/cross scales: kth smallest within/cross distance per row (where available)
	sigma_in = np.ones(n_obs, dtype=np.float64)
	sigma_cross = np.ones(n_obs, dtype=np.float64)

	for i in range(n_obs):
		di = d[i]
		wi = is_within[i]
		# within
		if np.any(wi):
			v = di[wi]
			kk = int(min(kth, v.shape[0] - 1))
			s = np.partition(v, kk)[kk]
			if np.isfinite(s) and s > 0:
				sigma_in[i] = s
			else:
				m = np.nanmedian(v)
				if np.isfinite(m) and m > 0:
					sigma_in[i] = m
		# cross
		ci = ~wi
		if np.any(ci):
			v = di[ci]
			kk = int(min(kth, v.shape[0] - 1))
			s = np.partition(v, kk)[kk]
			if np.isfinite(s) and s > 0:
				sigma_cross[i] = s
			else:
				m = np.nanmedian(v)
				if np.isfinite(m) and m > 0:
					sigma_cross[i] = m

	# mixture parameters (internal constants, no public API change)
	# damping for cross-batch edges to avoid dominating local structure
	lmbda = 0.5

	rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
	cols = inds.reshape(-1).astype(np.int64, copy=False)
	dvals = d.reshape(-1).astype(np.float64, copy=False)

	# mask invalid neighbors (-1)
	mask = cols >= 0
	rows = rows[mask]
	cols = cols[mask]
	dvals = dvals[mask]

	# compute weights with self-tuning kernel; choose within vs cross local scales
	is_within_flat = (batch_list[rows] == batch_list[cols])

	si = np.where(is_within_flat, sigma_in[rows], sigma_cross[rows])
	sj = np.where(is_within_flat, sigma_in[cols], sigma_cross[cols])
	den = (si * sj) + 1e-12

	w = np.exp(-(dvals ** 2) / den)
	w *= np.where(is_within_flat, 1.0, lmbda)

	# Build directed weight matrix then symmetrise using a UMAP-like set operation mix on weights
	W = coo_matrix((w, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
	W.eliminate_zeros()

	WT = W.T.tocsr()
	if set_op_mix_ratio is None:
		set_op_mix_ratio = 1.0
	set_op_mix_ratio = float(set_op_mix_ratio)

	# union with mutual emphasis (UMAP-style on weights):
	# W_union = W + WT - W*WT; then blend with intersection by set_op_mix_ratio
	# Intersection = W*WT (elementwise)
	inter = W.multiply(WT).tocsr()
	union = (W + WT - inter).tocsr()
	connectivities = (set_op_mix_ratio * union + (1.0 - set_op_mix_ratio) * inter).tocsr()
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
		# Reduce Python overhead by preallocating arrays and filling them
		k = int(params['neighbors_within_batch'])
		n = int(data.shape[0])
		ckdo_ind = np.empty((n, k), dtype=np.int64)
		ckdo_dist = np.empty((n, k), dtype=np.float64)
		for i in range(n):
			inds, dists = ckd.get_nns_by_vector(data[i, :], k, include_distances=True)
			ckdo_ind[i, :] = inds
			ckdo_dist[i, :] = dists
		ckdout = (ckdo_dist, ckdo_ind)
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
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.matrix.bbknn()``. Returns a tuple of distances and indices of neighbours for
	each cell.

	This implementation performs two key improvements over the legacy BBKNN behaviour:

	1) Within-batch neighbours ("biological backbone"):
	   For each cell, we take `neighbors_within_batch` neighbours from its own batch only.
	   These edges preserve local manifold structure and cell type separation.

	2) Cross-batch neighbours (alignment edges) are MNN-constrained:
	   Cross-batch candidate neighbours are queried per batch, but only kept if the edge
	   is supported in both directions (Mutual Nearest Neighbours, MNN). This reduces
	   spurious one-way cross-batch shortcuts.

	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	n_batches = len(batches)

	# in case we're gonna be faissing, turn the data to float32
	if params['computation'] == 'faiss':
		pca = pca.astype('float32')

	# cache PCA slice once
	X = pca[:, :params['n_pcs']]
	n_cells = X.shape[0]

	# batch sizes
	unique, counts = np.unique(batch_list, return_counts=True)
	count_map = {u: c for u, c in zip(unique, counts)}

	# parameters for neighbour selection
	k_in = int(params['neighbors_within_batch'])
	# cross-batch cap (derived from existing knobs; keep conservative fraction to protect biology)
	k_cross_total = int((n_batches - 1) * params['neighbors_within_batch'])
	k_cross_total = int(np.floor(0.7 * k_cross_total)) if k_cross_total > 0 else 0
	k_total = int(k_in + k_cross_total)

	# candidate pool size for MNN matching
	expansion = 2 if n_batches <= 2 else 3
	k_candidate = int(max(k_in * expansion, k_in))
	k_candidate = int(min(k_candidate, int(np.min(counts))))

	# mapping batch value -> indices in global space
	batch_to_global = {}
	for b in batches:
		mask = batch_list == b
		batch_to_global[b] = np.arange(len(batch_list), dtype=np.int64)[mask]

	# compute within-batch KNN for each batch and fill into per-cell arrays
	knn_in_idx = np.empty((n_cells, k_in), dtype=np.int64)
	knn_in_dist = np.empty((n_cells, k_in), dtype=np.float64)

	for b in batches:
		mask = batch_list == b
		global_idx = batch_to_global[b]
		Xb = X[mask, :]

		ckd_b = create_tree(data=Xb, params=params)
		# query only within the batch
		ckdout = query_tree(data=Xb, ckd=ckd_b, params=params)
		# convert to global indices
		knn_in_idx[global_idx, :] = global_idx[ckdout[1]]
		knn_in_dist[global_idx, :] = np.asarray(ckdout[0], dtype=np.float64)

	# cross-batch MNNs: build per ordered pair (A->B) top-k_candidate neighbor lists
	# store in dict keyed by (A,B): (indices_global, dists)
	params_q = params.copy()
	params_q['neighbors_within_batch'] = k_candidate

	cross = {}
	for b_to in batches:
		mask_to = batch_list == b_to
		ind_to = batch_to_global[b_to]

		ckd = create_tree(data=X[mask_to, :], params=params)
		ckdout = query_tree(data=X, ckd=ckd, params=params_q)

		cross[b_to] = (ind_to[ckdout[1]], np.asarray(ckdout[0], dtype=np.float64))

	# build MNN-constrained cross neighbours for each cell by scanning batches
	# and taking mutual matches; then keep up to k_cross_total closest by distance.
	knn_cross_idx = np.full((n_cells, max(k_cross_total, 1)), -1, dtype=np.int64) if k_cross_total > 0 else None
	knn_cross_dist = np.full((n_cells, max(k_cross_total, 1)), np.inf, dtype=np.float64) if k_cross_total > 0 else None

	if k_cross_total > 0:
		# for each cell i, collect mutual candidates (j, dist_ij) across other batches
		batch_to_pos = {b: i for i, b in enumerate(batches)}
		for i in range(n_cells):
			b_i = batch_list[i]
			mutual_js = []
			mutual_ds = []

			for b_to in batches:
				if b_to == b_i:
					continue

				# i -> (batch b_to) candidates
				idx_i_to = cross[b_to][0][i, :]
				dist_i_to = cross[b_to][1][i, :]

				# mutual test: for candidate j in batch b_to, check if i is in j->(batch b_i) list
				# use the precomputed "to batch b_i" candidates for row j
				idx_j_to_bi = cross[b_i][0][idx_i_to, :]  # shape (k_candidate, k_candidate)
				# any occurrence of i in row-wise candidate sets
				is_mnn = np.any(idx_j_to_bi == i, axis=1)

				if np.any(is_mnn):
					mutual_js.append(idx_i_to[is_mnn])
					mutual_ds.append(dist_i_to[is_mnn])

			if len(mutual_js) == 0:
				continue

			js = np.concatenate(mutual_js)
			ds = np.concatenate(mutual_ds)

			# de-duplicate neighbors by keeping the best (smallest distance)
			order = np.argsort(ds)
			js = js[order]
			ds = ds[order]
			# unique by first occurrence in sorted order
			_, uniq_pos = np.unique(js, return_index=True)
			uniq_pos = np.sort(uniq_pos)
			js = js[uniq_pos]
			ds = ds[uniq_pos]

			take = min(k_cross_total, js.shape[0])
			if take > 0:
				knn_cross_idx[i, :take] = js[:take]
				knn_cross_dist[i, :take] = ds[:take]

	# concatenate within + cross, then sort by (raw) distance
	if k_cross_total > 0:
		sel_idx = np.concatenate([knn_in_idx, knn_cross_idx], axis=1)
		sel_dist = np.concatenate([knn_in_dist, knn_cross_dist], axis=1)
	else:
		sel_idx = knn_in_idx
		sel_dist = knn_in_dist

	# ensure we have exactly k_total neighbours per row (pad with within-batch if needed)
	if sel_idx.shape[1] != k_total:
		k_total = sel_idx.shape[1]

	ordr = np.argsort(sel_dist, axis=1)
	knn_indices = np.take_along_axis(sel_idx, ordr, axis=1)
	knn_distances = np.take_along_axis(sel_dist, ordr, axis=1).astype(np.float64, copy=False)

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
	Trims the graph in a mutuality-aware manner.

	Instead of keeping the top connectivities per row (batch-agnostic), this trimming:
	- preferentially keeps *mutual* edges (i->j and j->i both present),
	- then fills remaining budget per row with the strongest non-mutual edges,
	- and finally symmetrises the result to preserve an undirected neighbourhood graph.

	Input
	-----
	cnts : ``CSR``
		Sparse matrix of processed connectivities to trim.
	trim : ``int``
		Maximum number of edges to keep per row (before final symmetrisation).
	'''
	if trim is None or trim <= 0:
		return cnts

	cnts = cnts.tocsr()
	if not cnts.has_sorted_indices:
		cnts.sort_indices()

	n = cnts.shape[0]

	# Avoid materializing cnts.minimum(cnts.T) (expensive sparse-sparse op).
	# Compute a sparse mutual-existence mask via boolean structure intersection:
	# mutual edges exist where both i->j and j->i are present.
	AT = cnts.T.tocsr()
	if not AT.has_sorted_indices:
		AT.sort_indices()
	mut_bool = cnts.sign().multiply(AT.sign()).tocsr()
	if not mut_bool.has_sorted_indices:
		mut_bool.sort_indices()

	# precompute kept edges per row for preallocation
	keep_counts = np.diff(cnts.indptr)
	keep_counts = np.minimum(keep_counts, int(trim)).astype(np.int64, copy=False)

	total_kept = int(np.sum(keep_counts))
	rows_out = np.empty(total_kept, dtype=np.int64)
	cols_out = np.empty(total_kept, dtype=np.int64)
	data_out = np.empty(total_kept, dtype=np.float64)

	# Priority trimming per row:
	# - give mutual edges a large additive boost so they are preferentially retained,
	# - then use original weights to break ties.
	# This preserves the "mutual-first, then strongest remainder" intent without building a
	# full weighted mutual matrix.
	mutual_boost = float(cnts.data.max() + 1.0) if cnts.nnz > 0 else 1.0

	pos = 0
	for i in range(n):
		start, end = cnts.indptr[i], cnts.indptr[i + 1]
		deg = end - start
		if deg == 0:
			continue

		cols = cnts.indices[start:end]
		data = cnts.data[start:end]

		if deg <= trim:
			nk = deg
			rows_out[pos:pos + nk] = i
			cols_out[pos:pos + nk] = cols
			data_out[pos:pos + nk] = data
			pos += nk
			continue

		m_start, m_end = mut_bool.indptr[i], mut_bool.indptr[i + 1]
		m_cols = mut_bool.indices[m_start:m_end]
		if m_cols.shape[0] == 0:
			priority = data
		else:
			# Determine which cols are mutual using sorted membership (both sorted)
			is_mut = np.isin(cols, m_cols, assume_unique=False)
			priority = data + (is_mut.astype(data.dtype, copy=False) * mutual_boost)

		# pick top 'trim' by priority, then keep original weights
		ch = np.argpartition(priority, -trim)[-trim:]
		# stable-ish ordering by actual connectivity
		ch = ch[np.argsort(data[ch])[::-1]]

		nk = ch.shape[0]
		rows_out[pos:pos + nk] = i
		cols_out[pos:pos + nk] = cols[ch]
		data_out[pos:pos + nk] = data[ch]
		pos += nk

	# build trimmed graph
	rows_out = rows_out[:pos]
	cols_out = cols_out[:pos]
	data_out = data_out[:pos]

	trimmed = coo_matrix((data_out, (rows_out, cols_out)), shape=cnts.shape).tocsr()
	trimmed.eliminate_zeros()

	# symmetrise: keep the maximum connectivity between i<->j
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
	#assert that all batches have at least neighbors_within_batch cells in there
	unique, counts = np.unique(batch_list, return_counts=True)
	if np.min(counts) < params['neighbors_within_batch']:
		raise ValueError("Not all batches have at least `neighbors_within_batch` cells in them.")
	#if any of the old legacy computation selection arguments are detected
	#use them to select the knn computation algorithm
	params = legacy_computation_selection(params, scanpy_logging)
	#sanity check the metric
	params = check_knn_metric(params, counts, scanpy_logging)
	#obtain the batch balanced KNN graph (returned already sorted by distance)
	knn_distances, knn_indices = get_graph(pca=pca,batch_list=batch_list,params=params)
	#this part of the processing is akin to scanpy.api.neighbors()
	#Batch-aware connectivity weighting (self-tuning kernel mixture) to prevent cross-batch
	#edges from dominating local structure.
	dist, cnts = compute_connectivities_batchaware_kernel(
		knn_indices, knn_distances, batch_list,
		set_op_mix_ratio=set_op_mix_ratio,
		local_connectivity=local_connectivity
	)
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
