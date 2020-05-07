import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from ...utils import l21_norm, matrix_norm, power_neg_half

class NDFS(object):
	"""
		Performs Nonnegative Discriminative Feature Selection algorithm.

		Parameters
		----------
		p : int
			Number of features to select.
		c : int, optional
			Amount of clusters to find.
		k : int, optional
			Amount of nearest neighbors to use while building the graph.
		alpha : float, optional
			Parameter in the objective function.
		beta : float, optional
			Regularization parameter in the objective function.
		gamma : float, optional
			Parameter in the objective function that controls the orthogonality condition.
		max_iterations : int, optional
			Maximum amount of iterations to perform.
		epsilon : positive float, optional
			Specifies the needed residual between the target functions from consecutive iterations. If the residual
			is smaller than epsilon, the algorithm is considered to have converged.
		sigma : float, optional
			Parameter for the weighting scheme.
		
		See Also
		--------
		http://www.nlpr.ia.ac.cn/2012papers/gjhy/gh27.pdf

		examples
		--------

	"""

	def __init__(self, p, c=5, k=5, alpha=1, beta=1, gamma=10e8, max_iterations=1000, epsilon=1e-5, sigma=1):
		self.p = p
		self.c = c
		self.k = k
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.max_iterations = max_iterations
		if epsilon < 0:
			raise ValueError("Epsilon should be positive, %d passed" % epsilon)
		self.epsilon = epsilon
		self.sigma = sigma

	def scheme(self, x1, x2):
		return np.exp(-np.linalg.norm(x1 - x2) ** 2 / self.sigma)

	def run(self, X, y=None):
		"""
			Fits filter

			Parameters
			----------
			X : numpy array, shape (n_samples, n_features)
				The training input samples.
			y : numpy array, shape (n_samples) or (n_samples, n_classes), optional
				The target values or their one-hot encoding that are used to compute F. If not present, a k-means clusterization algorithm is used.
				If present, n_classes should be equal to c.

			Returns
			----------
			W : array-like, shape (n_features, c)
				Feature weight matrix.

			See Also
			--------

			examples
			--------
			from ITMO_FS.filters.sparse import NDFS
			from sklearn.datasets import make_classification
			import numpy as np
			
			dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
			data, target = np.array(dataset[0]), np.array(dataset[1])
			model = NDFS(p=5, c=2)
			weights = model.run(data)
			print(model.feature_ranking(weights))
		"""

		n_samples, n_features = X.shape
		graph = NearestNeighbors(n_neighbors=self.p + 1, algorithm='ball_tree').fit(X).kneighbors_graph(X).toarray()
		graph = graph + graph.T

		indices = [[(i, j) for j in range(n_samples)] for i in range(n_samples)]
		func = np.vectorize(lambda xy: graph[xy[0]][xy[1]] * self.scheme(X[xy[0]], X[xy[1]]), signature='(1)->()')
		S = func(indices)

		A = np.diag(S.sum(axis=0))
		L = power_neg_half(A).dot(A - S).dot(power_neg_half(A))
		
		if y != None:
			if len(y.shape) == 2:
				Y = y
			else:
				Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
		else:
			Y = self.__run_kmeans(X)
		F = Y.dot(power_neg_half(Y.T.dot(Y)))
		D = np.eye(n_features)
		I = np.eye(n_samples)

		previous_target = 0
		for step in range(self.max_iterations):
			M = L + self.alpha * (I - X.dot(np.linalg.inv(X.T.dot(X) + self.beta * D)).dot(X.T))
			F = F * ((self.gamma * F) / (M.dot(F) + self.gamma * F.dot(F.T).dot(F)))
			W = np.linalg.inv(X.T.dot(X) + self.beta * D).dot(X.T.dot(F))
			diag = 2 * matrix_norm(W)
			diag[diag < 1e-10] = 1e-10  # prevents division by zero
			D = np.diag(1 / diag)

			target = np.trace(F.T.dot(L).dot(F)) + self.alpha * (np.linalg.norm(X.dot(W) - F) + self.beta * l21_norm(W))
			if step > 0 and abs(target - previous_target) < self.epsilon:
				break
			previous_target = target

		return W

	def feature_ranking(self, W):
		"""
			Calculate NDFS score for feature weight matrix.

			Parameters
			----------
			W : array-like, shape (n_features, c)
				Feature weight matrix.

			Returns
			-------
			indices : array-like, shape(p)
				Indices of p selected features.
		"""
		ndfs_score = matrix_norm(W)
		ranking = np.argsort(ndfs_score)[::-1]
		return ranking[:self.p]

	def __run_kmeans(self, X):
		n_samples, n_features = X.shape
		kmeans = KMeans(n_clusters=self.c, copy_x=True)
		kmeans.fit(X)
		labels = kmeans.labels_
		return OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
