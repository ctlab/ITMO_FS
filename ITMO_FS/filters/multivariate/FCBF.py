import numpy as np

from ...utils.information_theory import matrix_mutual_information
from ...utils import DataChecker, generate_features


# TODO X and y transformation for DataFrame support
class FCBFDiscreteFilter(DataChecker):
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

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> fcbf = FCBFDiscreteFilter()
        >>> print(fcbf.run(X, y))

    """

    def __init__(self):
        self.selected_features = None

    def fit(self, X, y, feature_names=None):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)

            y : numpy array, shape (n_samples, )

            feature_names : list of strings, optional
                In case you want to define feature names
            Returns
            ----------
            None
        """

        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        free_features = generate_features(X)
        self.feature_names = dict(zip(features, feature_names))
        self.selected_features = np.array([], dtype='object')
        # TODO Add exit of the loop when all differences are positive and are not updated
        #  (e.g. it happens when we get same max_index twice).
        max_index = -1
        while len(free_features) != 0:
            if max_index == np.argmax(matrix_mutual_information(X[:, free_features], y)):
                break
            max_index = np.argmax(matrix_mutual_information(X[:, free_features], y))
            self.selected_features = np.append(self.selected_features, max_index)
            relevance = matrix_mutual_information(X[:, free_features], y)
            redundancy = matrix_mutual_information(X[:, free_features], X[:, max_index])
            difference = relevance - redundancy
            free_features = np.delete(free_features, np.where(difference <= 0.)[0])
        self.selected_features = features[self.selected_features.astype(int)]

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            -------

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
            -------
            X dataset sliced with features selected by the filter
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
