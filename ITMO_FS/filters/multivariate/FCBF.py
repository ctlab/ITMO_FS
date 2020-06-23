import numpy as np

from ...utils.information_theory import matrix_mutual_information


# TODO X and y transformation for DataFrame support
class FCBFDiscreteFilter(object):
    """
        Creates FCBF (Fast Correlation Based filter) feature selection filter
        based on mutual information criteria for data with discrete features
        This filter finds best set of features by searching for a feature, which provides
        the most information about classification problem on given dataset at each step
        and then eliminating features which are less relevant than redundant

        Parameters
        ----------

        See Also
        --------
        https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf

        examples
        --------
        from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
        import numpy as np

        X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        fcbf = FCBFDiscreteFilter()
        print(fcbf.run(X, y))

    """

    def __init__(self):
        self.selected_features = None

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

        free_features = np.arange(0, X.shape[1], dtype=np.integer)
        self.selected_features = np.array([], dtype=np.integer)
        # TODO Add exit of the loop when all differences are positive and are not updated
        #  (e.g. it happens when we get same max_index twice).
        while free_features.size != 0:
            max_index = np.argmax(matrix_mutual_information(X[:, free_features], y))
            self.selected_features = np.append(self.selected_features, max_index)
            relevance = matrix_mutual_information(X[:, free_features], y)
            redundancy = matrix_mutual_information(X[:, free_features], X[:, max_index])
            difference = relevance - redundancy
            free_features = np.delete(free_features, np.where(difference <= 0.)[0])
        return self.selected_features
