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
	rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

	for i in range(knn_indices.shape[0]):
		for j in range(n_neighbors):
			if knn_indices[i, j] == -1:
				continue  # We didn't get the full knn for i
			if knn_indices[i, j] == i:
				val = 0.0
			else:
				val = knn_dists[i, j]

			rows[i * n_neighbors + j] = i
			cols[i * n_neighbors + j] = knn_indices[i, j]
			vals[i * n_neighbors + j] = val

	result = coo_matrix((vals, (rows, cols)),
									  shape=(n_obs, n_obs))
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
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.matrix.bbknn()``. Returns a tuple of distances and indices of neighbours for
	each cell.

	This implementation replaces heuristic "soft BBKNN" neighbour allocation with an explicit,
	per-cell *batch-fair* optimisation:

	- We query an expanded candidate pool from each batch (as before).
	- We keep distances in a shared metric space (no per-(cell,batch) median normalisation).
	- For each cell i, we solve a small entropic-regularised optimisation that minimises
	  distance while softly matching a target batch proportion q via a KL penalty.

	The optimisation is implemented via a lightweight fixed-point / Sinkhorn-style update over
	batch-level scaling factors, which is cheap especially for small number of batches.

	Batch-balance strength is controlled by params['bbknn_fairness_lambda'] (lambda).
	Entropy/temperature is controlled by params['bbknn_fairness_tau'] (tau).

	Target batch proportions q are set by params['bbknn_fairness_target']:
	- 'uniform' (default): q_b = 1/B
	- 'size'            : q_b proportional to batch sizes

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

	# total neighbours per cell expected downstream by UMAP
	k_total = int(params['neighbors_within_batch'] * n_batches)

	# query an expanded candidate pool per batch to allow optimisation flexibility
	k_candidate = int(max(params['neighbors_within_batch'] * 3, params['neighbors_within_batch']))

	# target batch proportions q_b
	unique, counts = np.unique(batch_list, return_counts=True)
	count_map = {u: c for u, c in zip(unique, counts)}
	batch_sizes = np.asarray([count_map[b] for b in batches], dtype=float)

	target_mode = params.get('bbknn_fairness_target', 'uniform')
	if target_mode not in ['uniform', 'size']:
		target_mode = 'uniform'
	if target_mode == 'size':
		q = batch_sizes / np.sum(batch_sizes)
	else:
		q = np.ones(n_batches, dtype=float) / float(n_batches)

	lam = float(params.get('bbknn_fairness_lambda', 2.0))  # 0 -> distance-only; larger -> more batch-fair
	tau = float(params.get('bbknn_fairness_tau', 1.0))      # temperature for exp(-d/tau)
	n_iter = int(params.get('bbknn_fairness_n_iter', 15))
	eps = 1e-12

	# store per-batch candidate neighbours (indices are global) and raw distances
	# shapes: (n_cells, k_candidate, n_batches)
	n_cells = pca.shape[0]
	cand_indices = np.zeros((n_cells, k_candidate, n_batches), dtype=np.int64)
	cand_dists = np.zeros((n_cells, k_candidate, n_batches), dtype=np.float64)

	# temporarily override neighbours_within_batch for querying candidates
	params_q = params.copy()
	params_q['neighbors_within_batch'] = k_candidate

	for b_ind in range(n_batches):
		batch_to = batches[b_ind]
		mask_to = batch_list == batch_to
		ind_to = np.arange(len(batch_list))[mask_to]

		ckd = create_tree(data=pca[mask_to, :params['n_pcs']], params=params)
		ckdout = query_tree(data=pca[:, :params['n_pcs']], ckd=ckd, params=params_q)

		# convert returned indices (relative to batch subset) to global indices
		cand_indices[:, :, b_ind] = ind_to[ckdout[1]]

		# keep distances in the shared metric space (no per-batch rescaling)
		cand_dists[:, :, b_ind] = np.asarray(ckdout[0], dtype=np.float64)

	# flatten candidates for selection
	flat_d = cand_dists.reshape(n_cells, k_candidate * n_batches)
	flat_i = cand_indices.reshape(n_cells, k_candidate * n_batches)
	flat_b = np.tile(np.arange(n_batches, dtype=np.int64), k_candidate)[None, :]
	flat_b = np.repeat(flat_b, n_cells, axis=0)

	# prepare final outputs
	knn_distances = np.zeros((n_cells, k_total), dtype=np.float64)
	knn_indices = np.zeros((n_cells, k_total), dtype=np.int64)

	# per-cell optimisation
	# We use the form: p_ij ∝ exp(-d_ij/tau) * exp(beta_{b(j)}),
	# where beta is updated so that batch-masses approximately match q under a KL penalty strength lam.
	# The update is:
	#   beta_b <- beta_b + lam * (log(q_b) - log(mass_b))
	# where mass_b = sum_{j in b} p_ij, after normalisation to sum(p)=k_total.
	for i in range(n_cells):
		d_i = flat_d[i]
		idx_i = flat_i[i]
		b_i = flat_b[i]

		# base affinities from distances
		# stabilise exponentials by shifting by min
		d_min = np.min(d_i) if d_i.shape[0] else 0.0
		base = np.exp(-(d_i - d_min) / max(tau, eps))

		# initialise beta
		beta = np.zeros(n_batches, dtype=np.float64)

		# fixed point iterations
		for _ in range(n_iter):
			w = base * np.exp(beta[b_i])
			s = np.sum(w)
			if s <= 0:
				break
			# normalise so sum_j p_ij = k_total
			p = (k_total * w) / (s + eps)

			# compute batch masses
			mass = np.bincount(b_i, weights=p, minlength=n_batches).astype(np.float64)
			mass = np.maximum(mass, eps)
			# update beta towards target proportions; stronger lam -> closer to q
			beta += lam * (np.log(q + eps) - np.log(mass / (np.sum(mass) + eps) + eps))

		# final weights
		w = base * np.exp(beta[b_i])
		# pick top-k_total by weight (equivalently dual-adjusted distance)
		if w.shape[0] > k_total:
			choose = np.argpartition(w, -k_total)[-k_total:]
		else:
			choose = np.arange(w.shape[0])

		# order chosen neighbours by true distance (stable downstream)
		order = np.argsort(d_i[choose])
		choose = choose[order]

		knn_distances[i, :len(choose)] = d_i[choose]
		knn_indices[i, :len(choose)] = idx_i[choose]

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
	n = cnts.shape[0]

	# mutual edges mask (for nonzeros): compute intersection with transpose sparsity
	mut = cnts.minimum(cnts.T).tocsr()

	# build trimmed result row-by-row
	rows = []
	cols = []
	data = []

	for i in range(n):
		# mutual candidates
		m_start, m_end = mut.indptr[i], mut.indptr[i + 1]
		m_cols = mut.indices[m_start:m_end]
		m_data = mut.data[m_start:m_end]

		# all candidates
		c_start, c_end = cnts.indptr[i], cnts.indptr[i + 1]
		c_cols = cnts.indices[c_start:c_end]
		c_data = cnts.data[c_start:c_end]

		if c_cols.shape[0] <= trim:
			rows.append(np.full(c_cols.shape[0], i, dtype=np.int64))
			cols.append(c_cols.astype(np.int64, copy=False))
			data.append(c_data.astype(np.float64, copy=False))
			continue

		keep_cols = []
		keep_data = []

		# 1) keep strongest mutual edges up to trim
		if m_cols.shape[0] > 0:
			if m_cols.shape[0] > trim:
				choose = np.argpartition(m_data, -trim)[-trim:]
				order = np.argsort(m_data[choose])[::-1]
				choose = choose[order]
			else:
				choose = np.argsort(m_data)[::-1]
			keep_cols.extend(m_cols[choose].tolist())
			keep_data.extend(m_data[choose].tolist())

		# 2) fill remainder with strongest non-mutual edges
		remaining = trim - len(keep_cols)
		if remaining > 0:
			# mask out already kept cols
			if len(keep_cols) > 0:
				mask = ~np.isin(c_cols, np.asarray(keep_cols, dtype=c_cols.dtype))
				f_cols = c_cols[mask]
				f_data = c_data[mask]
			else:
				f_cols = c_cols
				f_data = c_data
			if f_cols.shape[0] > 0:
				if f_cols.shape[0] > remaining:
					choose = np.argpartition(f_data, -remaining)[-remaining:]
					order = np.argsort(f_data[choose])[::-1]
					choose = choose[order]
				else:
					choose = np.argsort(f_data)[::-1]
				keep_cols.extend(f_cols[choose].tolist())
				keep_data.extend(f_data[choose].tolist())

		rows.append(np.full(len(keep_cols), i, dtype=np.int64))
		cols.append(np.asarray(keep_cols, dtype=np.int64))
		data.append(np.asarray(keep_data, dtype=np.float64))

	rows = np.concatenate(rows) if len(rows) else np.asarray([], dtype=np.int64)
	cols = np.concatenate(cols) if len(cols) else np.asarray([], dtype=np.int64)
	data = np.concatenate(data) if len(data) else np.asarray([], dtype=np.float64)

	trimmed = coo_matrix((data, (rows, cols)), shape=cnts.shape).tocsr()
	trimmed.eliminate_zeros()

	# symmetrise: keep the maximum connectivity between i<->j
	trimmed = trimmed.maximum(trimmed.T).tocsr()
	trimmed.eliminate_zeros()
	return trimmed

def bbknn(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None,
		  computation='annoy', annoy_n_trees=10, pynndescent_n_neighbors=30,
		  pynndescent_random_state=0, metric='euclidean', set_op_mix_ratio=1,
		  local_connectivity=1, approx=None, use_annoy=None, use_faiss=None,
		  scanpy_logging=False,
		  bbknn_fairness_lambda=2.0, bbknn_fairness_tau=1.0, bbknn_fairness_target='uniform',
		  bbknn_fairness_n_iter=15,
		  bbknn_cross_batch_gamma=1.0):
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

	# Batch-aware connectivity reweighting ("two-temperature kernel"):
	# down/up-weight cross-batch edges by gamma (<=1 dampens cross-batch connections)
	gamma = float(params.get('bbknn_cross_batch_gamma', 1.0))
	if gamma != 1.0:
		# ensure batch_list is aligned and stringified (already done above)
		batch_arr = np.asarray(batch_list)
		cnts = cnts.tocsr(copy=True)
		for i in range(cnts.shape[0]):
			start, end = cnts.indptr[i], cnts.indptr[i + 1]
			if start == end:
				continue
			cols = cnts.indices[start:end]
			cross = batch_arr[cols] != batch_arr[i]
			if np.any(cross):
				cnts.data[start:end][cross] *= gamma
		# keep graph symmetric/undirected after reweighting
		cnts = cnts.maximum(cnts.T).tocsr()
		cnts.eliminate_zeros()
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
