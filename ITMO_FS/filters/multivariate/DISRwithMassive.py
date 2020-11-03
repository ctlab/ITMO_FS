import numpy as np

from ITMO_FS.utils.information_theory import entropy
from ITMO_FS.utils.information_theory import mutual_information
from ...utils import BaseTransformer, generate_features

def _complementarity(x_i, x_j, y):
    return entropy(x_i) + entropy(x_j) + entropy(y) - entropy(list(zip(x_i, x_j))) - \
           entropy(list(zip(x_i, y))) - entropy(list(zip(x_j, y))) + entropy(list(zip(x_i, x_j, y)))


def _chained_information(x_i, x_j, y):
    return mutual_information(x_i, y) + mutual_information(x_j, y) + _complementarity(x_i, x_j, y)

class DISRWithMassive(BaseTransformer):
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
        n_features : int
            Number of features to select.

        Notes
        -----
        For more details see `this paper
        <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.318.6576&rep=rep1&type=pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import DISRWithMassive
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> disr = DISRWithMassive(3)
        >>> disr.fit_transform(X, y)
        array([[1, 2, 1],
               [2, 2, 2],
               [1, 3, 3],
               [3, 1, 4],
               [4, 4, 5]])
    """

    def __init__(self, n_features):
        self.n_features = n_features

    def __count_weight(self, i):
        return np.sum(2 * self._vertices[i] * np.multiply(self._edges[i], self._vertices))

    def _fit(self, X, y):
        """
            Fits filter

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

            Returns
            -------
            None
        """

        if self.n_features > self.n_features_:
            raise ValueError("Cannot select %d features with n_features = %d" % (self.n_features, self.n_features_))
            
        free_features = np.array([], dtype='object')
        self.selected_features_ = generate_features(X)
        self._vertices = np.ones(self.n_features_)
        self._edges = np.zeros((self.n_features_, self.n_features_))
        for i in range(self.n_features_):
            for j in range(self.n_features_):
                entropy_pair = entropy(list(zip(X[:, i], X[:, j])))
                if entropy_pair != 0.:
                    self._edges[i][j] = _chained_information(X[:, i], X[:, j], y) / entropy_pair

        # TODO apply vectorize to selected_features and not arange(n_features)?
        while len(self.selected_features_) != self.n_features:
            min_index = np.argmin(np.vectorize(lambda x: self.__count_weight(x))(self.selected_features_))
            self._vertices[min_index] = 0
            free_features = np.append(free_features, min_index)
            self.selected_features_ = np.delete(self.selected_features_, min_index)

        change = True
        while change:
            change = False
            swap_index = (-1, -1)
            max_difference = 0
            for i in range(len(free_features)):
                for j in range(len(self.selected_features_)):
                    temp_difference = self.__count_weight(free_features[i]) - self.__count_weight(
                        self.selected_features_[j])
                    if temp_difference > max_difference:
                        max_difference = temp_difference
                        swap_index = (i, j)
            if max_difference > 0:
                change = True
                new_selected, new_free = swap_index
                free_features = np.append(free_features, new_free)
                free_features = np.delete(free_features, new_selected)
                self.selected_features_ = np.append(self.selected_features_, new_selected)
                self.selected_features_ = np.delete(self.selected_features_, new_free)
