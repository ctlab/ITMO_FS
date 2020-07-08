import numpy as np

from ITMO_FS.utils.information_theory import entropy
from ITMO_FS.utils.information_theory import mutual_information
from ...utils import DataChecker, generate_features

def _complementarity(x_i, x_j, y):
    return entropy(x_i) + entropy(x_j) + entropy(y) - entropy(list(zip(x_i, x_j))) - \
           entropy(list(zip(x_i, y))) - entropy(list(zip(x_j, y))) + entropy(list(zip(x_i, x_j, y)))


def _chained_information(x_i, x_j, y):
    return mutual_information(x_i, y) + mutual_information(x_j, y) + _complementarity(x_i, x_j, y)


# TODO X and y transformation for DataFrame support
class DISRWithMassive(DataChecker):
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
        self.n_features = None
        self._vertices = None
        self._edges = None
        self.selected_features = None

    def __count_weight(self, i):
        temp_difference = 0
        for j in range(self.n_features):
            temp_difference += self._edges[i][j] * self._vertices[i] * self._vertices[j]
            temp_difference += self._edges[j][i] * self._vertices[i] * self._vertices[j]
        return temp_difference
        #TODO: this was supposed to be an optimization
        #return 2 * self._vertices[i] * np.multiply(self._edges[i], self._vertices) 

    def fit(self, X, y, feature_names=None):
        """
            Fits filter

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ----------
            selected_features : numpy array
                selected pool of features

        """
        
        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        self.feature_names = dict(zip(features, feature_names))
        self.n_features = X.shape[1]
        if self.expected_size is None:
            self.expected_size = self.n_features // 3
        free_features = np.array([], dtype='object')
        self.selected_features = generate_features(X)
        self._vertices = np.ones(self.n_features)
        self._edges = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            for j in range(self.n_features):
                entropy_pair = entropy(list(zip(X[:, i], X[:, j])))
                if entropy_pair != 0.:
                    self._edges[i][j] = _chained_information(X[:, i], X[:, j], y) / entropy_pair

        # TODO apply vectorize to selected_features and not arange(n_features)?
        while self.selected_features.size != self.expected_size:
            min_index = np.argmin(np.vectorize(lambda x: self.__count_weight(x))(self.selected_features))
            self._vertices[min_index] = 0
            free_features = np.append(free_features, min_index)
            self.selected_features = np.delete(self.selected_features, min_index)

        change = True
        while change:
            change = False
            swap_index = (-1, -1)
            max_difference = 0
            for i in range(len(free_features)):
                for j in range(len(self.selected_features)):
                    temp_difference = self.__count_weight(free_features[i]) - self.__count_weight(
                        self.selected_features[j])
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
        self.selected_features = features[self.selected_features]

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            ------

            Transformed 2D numpy array

        """

        if type(X) is np.ndarray:
            return X[:, self.selected_features.astype(int)]
        else:
            return X[self.selected_features]

    def fit_transform(self, X, y, feature_names=None):
        """
            Fits the filter and transforms given dataset X.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ------

            X dataset sliced with features selected by the filter
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
