import numpy as np
from scipy.linalg import eigh
from ...utils import knn, l21_norm, matrix_norm

class UDFS(object):
	"""
		Performs Unsupervised Discriminative Feature Selection algorithm.

		Parameters
		----------
		p : int
			Number of features to select.
		c : int, optional
			Amount of clusters to find.
		k : int, optional
			Amount of nearest neighbors to use while building the graph.
		gamma : float, optional
			Regularization term in the target function.
		l : float, optional
			Parameter that controls the invertibility of the matrix used in computing of B.
		max_iterations : int, optional
			Maximum amount of iterations to perform.
		epsilon : positive float, optional
			Specifies the needed residual between the target functions from consecutive iterations. If the residual
			is smaller than epsilon, the algorithm is considered to have converged.
		
		See Also
		--------
		https://www.ijcai.org/Proceedings/11/Papers/267.pdf

		examples
		--------

	"""

	def __init__(self, p, c=5, k=5, gamma=1, l=1e-6, max_iterations=1000, epsilon=1e-5):
		self.p = p
		self.c = c
		self.k = k
		self.gamma = gamma
		self.l = l
		self.max_iterations = max_iterations
		if epsilon < 0:
			raise ValueError("Epsilon should be positive, %d passed" % epsilon)
		self.epsilon = epsilon

	def run(self, X, y=None):
		"""
			Fits filter

			Parameters
			----------
			X : numpy array, shape (n_samples, n_features)
				The training input samples.
			y : numpy array, optional
				The target values (ignored).

			Returns
			----------
			W : array-like, shape (n_features, c)
				Feature weight matrix.

			See Also
			--------

			examples
			--------
			from ITMO_FS.filters.sparse import UDFS
			from sklearn.datasets import make_classification
			import numpy as np
			
			dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
			data, target = np.array(dataset[0]), np.array(dataset[1])
			model = UDFS(p=5, c=2)
			weights = model.run(data)
			print(model.feature_ranking(weights))
		"""

		def construct_S(arr):
			S = np.zeros((n_samples, self.k + 1))
			for idx in range(self.k + 1):
				S[arr[idx], idx] = 1
			return S

		n_samples, n_features = X.shape
		indices = list(range(n_samples))
		H = np.eye(self.k + 1) - np.ones((self.k + 1, self.k + 1)) / (self.k + 1)
		I = np.eye(self.k + 1)
		neighbors = np.vectorize(lambda idx: np.append([idx], knn(X, y, idx, self.k)), signature='()->(1)')(indices)
		X_centered = np.apply_along_axis(lambda arr: X[arr].T.dot(H), 1, neighbors)
		S = np.apply_along_axis(lambda arr: construct_S(arr), 1, neighbors)
		B = np.vectorize(lambda idx: np.linalg.inv(X_centered[idx].T.dot(X_centered[idx]) + self.l * I), signature='()->(1,1)')(indices)
		Mi = np.vectorize(lambda idx: S[idx].dot(H).dot(B[idx]).dot(H).dot(S[idx].T), signature='()->(1,1)')(indices)
		M = X.T.dot(Mi.sum(axis=0)).dot(X)

		D = np.eye(n_features)
		previous_target = 0
		for step in range(self.max_iterations):
			P = M + self.gamma * D
			_, W = eigh(a=P, eigvals=(0, self.c - 1))
			diag = 2 * matrix_norm(W)
			diag[diag < 1e-10] = 1e-10  # prevents division by zero
			D = np.diag(1 / diag)

			target = np.trace(W.T.dot(M).dot(W)) + self.gamma * l21_norm(W)
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
		udfs_score = matrix_norm(W)
		ranking = np.argsort(udfs_score)[::-1]
		return ranking[:self.p]
