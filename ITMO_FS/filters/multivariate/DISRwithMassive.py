import numpy as np
from ITMO_FS.utils.information_theory import mutual_information
from ITMO_FS.utils.information_theory import calc_entropy

class DISRWithMassive(object):

	"""
		Creates DISR (Double Input Symmetric Relevance) feature selection filter
		based on kASSI criterin for feature selection 
		which aims at maximizing the mutual information avoiding, meanwhile, large multivariate density estimation.
		Its a kASSI criterion with approximation of the information of a set of variables 
		by counting average information of subset on combination of two features.
		This formulation thus deals with feature complementarity up to order two 
		by preserving the same computational complexity of the MRMR and CMIM criteria
		The DISR calculation is done using graph based solution.

		Parameters
		----------
		expected_size : int
        	Expected size of subset of features.
		
		See Also
		--------
		http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.318.6576&rep=rep1&type=pdf

		examples
		--------
		from ITMO_FS.filters.multivariate import DISRWithMassive
		import numpy as np
		
		X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
		y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
		disr = DISRWithMassive(3)
		print(disr.run(X, y))

	"""

	def __init__(self, expected_size=None):
		self.expected_size = expected_size

	def __complementarity(self, x_i, x_j, y):
		return calc_entropy(x_i) + calc_entropy(x_j) + calc_entropy(y) - calc_entropy(list(zip(x_i, x_j))) - \
			calc_entropy(list(zip(x_i, y))) - calc_entropy(list(zip(x_j, y))) + calc_entropy(list(zip(x_i, x_j, y)))

	def __chained_information(self, x_i, x_j, y):
		return mutual_information(x_i, y) + mutual_information(x_j, y) + self.__complementarity(x_i, x_j, y)

	def __count_weight(self, i):
		return 2 * self.vertices[i] * np.multiply(self.edges[i], self.vertices)

	def run(self, X, y):
		"""
			Fits filter

			Parameters
			----------
			X : numpy array, shape (n_samples, n_features)

			y : numpy array, shape (n_samples, )

			Returns
			----------
			selected_features : numpy array
				selected pool of features

		"""


		self.n_features = X.shape[1]
		if self.expected_size == None:
			self.expected_size = self.n_features / 3
		free_features = np.array([], dtype=np.integer)
		self.selected_features = np.arange(self.n_features, dtype=np.integer)
		self.vertices = np.ones(self.n_features)
		self.edges = np.zeros((self.n_features, self.n_features))
		for i in range(self.n_features):
			for j in range(self.n_features):
				entropy = calc_entropy(list(zip(X[:, i], X[:, j])))
				if entropy != 0.:
					self.edges[i][j] = self.__chained_information(X[:, i], X[:, j], y) / entropy

		while(self.selected_features.size != self.expected_size):
			min_index = np.argmin(np.vectorize(lambda i: self.__count_weight(i))(np.arange(self.n_features)))
			self.vertices[min_index] = 0
			free_features = np.append(free_features, min_index)
			self.selected_features = np.delete(self.selected_features, min_index)

		change = True
		while(change):
			change = False
			swap_pair = (-1, -1)
			max_difference = 0
			for i in range(len(free_features)):
				for j in range(len(self.selected_features)):
					temp_difference = self.__count_weight(free_features[i]) - self.__count_weight(self.selected_features[j])
					if temp_difference > max_difference:
						max_difference = temp_difference
						swap_index = (i, j)
			if max_difference > 0:
				change = True
				new_selected, new_free = swap_index
				free_features = np.append(free_features, new_free)
				free_features = np.delete(free_features, new_selected)
				self.selected_features = np.append(self.selected_features, new_selected)
				self.selected_features = np.delete(self.selected_features, new_free)

		return self.selected_features
