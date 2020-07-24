import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ...utils import l21_norm, matrix_norm

class RFS(object):
	"""
		Performs the Robust Feature Selection via Joint L2,1-Norms Minimization algorithm.

		Parameters
		----------
		p : int
			Number of features to select.
		gamma : float, optional
			Regularization parameter.
		max_iterations : int, optional
			Maximum amount of iterations to perform.
		epsilon : positive float, optional
			Specifies the needed residual between the target functions from consecutive iterations. If the residual
			is smaller than epsilon, the algorithm is considered to have converged.

		See Also
		--------
		https://papers.nips.cc/paper/3988-efficient-and-robust-feature-selection-via-joint-l21-norms-minimization.pdf

		Examples
		--------
		>>> from ITMO_FS.filters.sparse import RFS
		>>> from sklearn.datasets import make_classification
		>>> import numpy as np
		>>> dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
		>>> data, target = np.array(dataset[0]), np.array(dataset[1])
		>>> model = RFS(gamma=15, epsilon=1e-12)
		>>> print(model.run(data, target))
	"""

	def __init__(self, p, gamma=1, max_iterations=1000, epsilon=1e-5):
		self.p = p
		self.gamma = gamma
		self.max_iterations = max_iterations
		if epsilon < 0:
			raise ValueError("Epsilon should be positive, %d passed" % epsilon)
		self.epsilon = epsilon

	def run(self, X, y):
		"""
			Fits the algorithm.

			Parameters
			----------
			X : array-like, shape (n_samples, n_features)
				The training input samples.
			y : array-like, shape (n_samples) or (n_samples, n_classes)
				The target values or their one-hot encoding.

			Returns
			----------
			W : array-like, shape (n_features, n_classes)
				Feature weight matrix.

			See Also
			--------

			Examples
			--------

		"""

		if len(y.shape) == 2:
			Y = y
		else:
			Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

		n_samples, n_features = X.shape
		A = np.append(X, self.gamma * np.eye(n_samples), axis=1)
		D = np.eye(n_samples + n_features)

		previous_target = 0
		for step in range(self.max_iterations):
			D_inv = np.linalg.inv(D)
			U = D_inv.dot(A.T).dot(np.linalg.inv(A.dot(D_inv).dot(A.T))).dot(Y)
			U = np.dot(np.dot(np.dot(D_inv, A.T), np.linalg.inv(np.dot(np.dot(A, D_inv), A.T))), Y)
			diag = 2 * matrix_norm(U)
			diag[diag < 1e-10] = 1e-10  # prevents division by zero
			D = np.diag(1 / diag)

			target = l21_norm(X.dot(U[:n_features]) - Y) + self.gamma * l21_norm(U[:n_features])
			if step > 0 and abs(target - previous_target) < self.epsilon:
				break
			previous_target = target

		return U[:n_features]


	def feature_ranking(self, W):
		"""
			Calculate the RFS score for a feature weight matrix.

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
