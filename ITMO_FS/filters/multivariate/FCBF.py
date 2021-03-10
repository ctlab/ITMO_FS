import numpy as np

from ...utils.information_theory import matrix_mutual_information
from ...utils import BaseTransformer, generate_features


class FCBFDiscreteFilter(BaseTransformer):
    """
    Creates FCBF (Fast Correlation Based filter) feature selection filter
    based on mutual information criteria for data with discrete features
    This filter finds best set of features by searching for a feature,
    which provides the most information about classification problem on
    given dataset at each step and then eliminating features which are less
    relevant than redundant

        Parameters
        ----------

        Notes ----- For more details see `this paper
        <https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> fcbf = FCBFDiscreteFilter()
        >>> fcbf.fit_transform(X, y)
        array([[1],
               [2],
               [3],
               [4],
               [5]])
    """

    def __init__(self):
        super().__init__()

    def _fit(self, x, y):
        """
            Fits filter

            Parameters
            ----------
            x : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.

            Returns
            -------
            None
        """

        free_features = generate_features(x)
        self.selected_features_ = np.array([], dtype='int')
        # TODO Add exit of the loop when all differences are positive and are not updated
        #  (e.g. it happens when we get same max_index twice).
        max_index = -1
        while len(free_features) != 0:
            if max_index == np.argmax(
                    matrix_mutual_information(x[:, free_features], y)):
                break
            max_index = np.argmax(
                matrix_mutual_information(x[:, free_features], y))
            self.selected_features_ = np.append(
                self.selected_features_, max_index)
            relevance = matrix_mutual_information(x[:, free_features], y)
            redundancy = matrix_mutual_information(
                x[:, free_features], x[:, max_index])
            difference = relevance - redundancy
            free_features = np.delete(
                free_features, np.where(
                    difference <= 0.)[0])
