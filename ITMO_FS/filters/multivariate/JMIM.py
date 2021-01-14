import numpy as np
from ITMO_FS.utils.information_theory import matrix_mutual_information, joint_mutual_information as JMI,\
                                            symmetrical_relevance as SR
from ITMO_FS.utils import DataChecker, generate_features


class JMIM(DataChecker):
    """
        Creates JMIM (Joint Mutual Information Maximisation) feature selection filter
        based on joint mutual information criterion for data with discrete features.
        This filter finds best set of features by maximasing joint mutual information for
        a candidate feature on its combination with target variable and each of the already
        selected features separately.
        Normalized JMIM uses normalised joint mutual information as a criterion.

        Parameters
        ----------

        Notes
        -----
        For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0957417415004674>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import JMIM
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> jmim = JMIM()
        >>> jmim.fit_transform(X, y)
        array([[1],
               [2],
               [3],
               [4],
               [5]])
    """


    def __init__(self, n_features_to_keep=10, normalized=False):
        self.selected_features = None
        self.n_features_to_keep = n_features_to_keep
        self.normalized = normalized


    def __calc_jmi(self, fi, fs, y):
        if self.normalized:
            jmi_scores = SR(fi, fs, y)
        else:
            jmi_scores = JMI(fi, fs, y)
        return jmi_scores


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
            -------
            None
        """

        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        self.free_features = generate_features(X)
        self.feature_names = dict(zip(features, feature_names))

        mi = matrix_mutual_information(X, y)
        self.selected_scores = np.array([max(mi)], dtype='float')
        selected_index = np.argmax(mi)
        self.selected_features = np.array([selected_index], dtype='int')
        self.free_features = self.free_features[self.free_features != selected_index]

        max_features = min(self.free_features.size, self.n_features_to_keep) - 1
        for _ in range(max_features):
            jmi_scores = np.array(
                [min([self.__calc_jmi(X[:, fi], X[:, fs], y) for fs in self.selected_features])
                                                                for fi in self.free_features])
            best_jmi = jmi_scores.max()
            selected_index = self.free_features[np.argmax(jmi_scores)]
            self.selected_features = np.append(self.selected_features, selected_index)
            self.free_features = self.free_features[self.free_features != selected_index]
            self.selected_scores = np.append(self.selected_scores, best_jmi)

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
            return X[:, self.selected_features]
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