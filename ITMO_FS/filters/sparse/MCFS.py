import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Lars
from scipy.linalg import eigh

class MCFS(object):
	"""
		Performs Unsupervised Feature Selection for Multi-Cluster Data algorithm.

		Parameters
		----------
		d : int
			Number of features to select.
		k : int, optional
			Amount of clusters to find.
		p : int, optional
			Amount of nearest neighbors to use while building the graph.
		scheme : str, either '0-1', 'heat' or 'dot', optional
			Weighting scheme to use while building the graph.
		sigma : float, optional
			Parameter for heat weighting scheme. Ignored if scheme is not 'heat'.
		
		See Also
		--------
		http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2010_KDD-MCFS.pdf

		examples
		--------
		from ITMO_FS.filters.unsupervised.trace_ratio_laplacian import TraceRatioLaplacian
		from sklearn.datasets import make_classification

		x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
		tracer = TraceRatioLaplacian(10)
		print(tracer.run(x, y)[0])

	"""

	def scheme_01(self, x1, x2):
		return 1

	def scheme_heat(self, x1, x2):
		return np.exp(-np.linalg.norm(x1 - x2) ** 2 / self.sigma)

	def scheme_dot(self, x1, x2):
		return (x1 / np.linalg.norm(x1)).dot(x2 / np.linalg.norm(x2))

	def __init__(self, d, k=5, p=5, scheme='dot', sigma=1):
		if scheme not in ['0-1', 'heat', 'dot']:
			raise KeyError('scheme should be either 0-1, heat or dot; %r passed' % scheme)
		if scheme == '0-1':
			self.scheme = self.scheme_01
		elif scheme == 'heat':
			self.scheme = self.scheme_heat
		else:
			self.scheme = self.scheme_dot
		self.d = d
		self.k = k
		self.p = p
		self.sigma = sigma

	def run(self, X, y):
		"""
			Fits filter

			Parameters
			----------
			X : numpy array, shape (n_samples, n_features)
			  The training input samples.
			y : numpy array, shape (n_samples)
			  The target values.

			Returns
			----------
			W : array-like, shape (n_features, n_classes)
				Feature weight matrix.

			See Also
			--------

			examples
			--------
			from ITMO_FS.filters.sparse import MCFS
			from sklearn.datasets import make_classification
			import numpy as np
			
			dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
			data, target = np.array(dataset[0]), np.array(dataset[1])
			model = MCFS(d=5, k=2, scheme='heat')
			weights = model.run(data, target)
			print(model.feature_ranking(weights))

		"""
		n_samples, n_features = X.shape
		graph = NearestNeighbors(n_neighbors=self.p + 1, algorithm='ball_tree').fit(X).kneighbors_graph(X).toarray()
		graph = graph + graph.T

		np.fill_diagonal(graph, 0)
		indices = [[(i, j) for j in range(n_samples)] for i in range(n_samples)]
		func = np.vectorize(lambda xy: graph[xy[0]][xy[1]] * self.scheme(X[xy[0]], X[xy[1]]), signature='(1)->()')
		W = func(indices)

		D = np.diag(W.sum(axis=0))
		L = D - W
		eigvals, Y = eigh(type=1, a=L, b=D, eigvals=(0, self.k))

		weights = np.zeros((n_features, self.k))
		for i in range(self.k):
			clf = Lars(n_nonzero_coefs=self.d)
			clf.fit(X, Y[:, i])
			weights[:, i] = clf.coef_

		return weights

	def feature_ranking(self, W):
		"""
			Calculate MCFS score for feature weight matrix.

			Parameters
			----------
			W : array-like, shape (n_features, n_classes)
				Feature weight matrix.

			Returns
			-------
			indices : array-like, shape(d)
				Indices of d selected features.
		"""
		mcfs_score = W.max(axis=1)
		ranking = np.argsort(mcfs_score)[::-1]
		return ranking[:self.d]