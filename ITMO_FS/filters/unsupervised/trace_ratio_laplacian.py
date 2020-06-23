import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *

#TODO requests changes for MultivariateFilter to be used there
class TraceRatioLaplacian(object):
	"""
		Creates TraceRatio(similarity based) feature selection filter
		performed in unsupervised way, i.e laplacian version

		Parameters
		----------
		n_selected_features : int
		    Amount of features to filter
		k : int, 
			number of neighbours to use for knn
		t : int,
			constant for kernel function calculation
		Note: in laplacian case only
		in fisher it uses label similarity, i.e if both samples belong to same class 
		See Also
		--------
		https://www.aaai.org/Papers/AAAI/2008/AAAI08-107.pdf

		examples
		--------
		from ITMO_FS.filters.unsupervised.trace_ratio_laplacian import TraceRatioLaplacian
		from sklearn.datasets import make_classification

		x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
		tracer = TraceRatioLaplacian(10)
		print(tracer.run(x, y)[0])


	"""
	def __init__(self, n_selected_features, k=5, t=1):
		self.n_selected_features = n_selected_features
		self.k = k
		self.t = t

	def run(self, X, y):
		"""
			Fits filter

			Parameters
			----------
			X : numpy array, shape (n_samples, n_features)
			  The training input samples
			y : numpy array, shape (n_samples, )
			  The target values

			Returns
			----------
			feature_indices : numpy array
				array of feature indices in X

			See Also
			--------

			examples
			--------

		"""
		
		n_samples = X.shape[0]
		Distances = pairwise_distances(X)
		Distances **= 2
		Distances_NN = np.sort(Distances, axis = 1)[:, 0:self.k + 1]
		Indices_NN = np.argsort(Distances, axis = 1)[:, 0:self.k + 1]
		Kernel = np.exp(-Distances_NN / self.t)
		joined_distances = np.ravel(Kernel)
		indices_axis_one = np.ravel(Indices_NN)
		indices_axis_zero = np.repeat(np.arange(n_samples), self.k + 1)
		A_within = csc_matrix((joined_distances, (indices_axis_zero, indices_axis_one)), shape = (n_samples, n_samples))
		A_within = A_within - A_within.multiply(A_within.T > A_within) + A_within.T.multiply(A_within.T > A_within)
		D_within = np.diag(np.ravel(A_within.sum(1))) # check correctness
		L_within = D_within - A_within
		A_between = D_within.dot(np.ones((n_samples, n_samples))).dot(D_within) / np.sum(D_within)
		D_between = np.diag(A_between.sum(1))
		L_between = D_between - A_between

		L_within = (L_within.T + L_within) / 2
		L_between = (L_between.T + L_between) / 2
		E = X.T.dot(L_within).dot(X)
		B = X.T.dot(L_between).dot(X)
		E = (E.T + E) / 2
		B = (B.T + B) / 2

		# we need only diagonal elements for trace calculation
		e = np.absolute(np.diag(E))
		b = np.absolute(np.diag(B))
		b[b == 0] = 1e-14
		features_indices = np.argsort(np.divide(b, e))[::-1][0:self.n_selected_features]
		lam = np.sum(b[features_indices])/np.sum(e[features_indices])
		prev_lam = 0
		while (lam - prev_lam >= 1e-3):
			score = b - lam * e
			features_indices = np.argsort(score)[::-1][0:self.n_selected_features]
			prev_lam = lam
			lam = np.sum(b[features_indices])/np.sum(e[features_indices])
		return features_indices, score, lam
