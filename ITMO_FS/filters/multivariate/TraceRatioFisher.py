import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
from ...utils import DataChecker, generate_features


# TODO X and y transformation for DataFrame support
# TODO requests changes for MultivariateFilter to be used there
class TraceRatioFisher(DataChecker):
    """
        Creates TraceRatio(similarity based) feature selection filter
        performed in supervised way, i.e fisher version

        Parameters
        ----------
        n_selected_features : int
            Amount of features to filter

        See Also
        --------
        https://www.aaai.org/Papers/AAAI/2008/AAAI08-107.pdf

        examples
        --------
        from ITMO_FS.filters.multivariate.trace_ratio_fisher import TraceRatioFisher
        from sklearn.datasets import make_classification

        x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
        tracer = TraceRatioFisher(10)
        print(tracer.run(x, y)[0])


    """

    def __init__(self, n_selected_features):
        self.n_selected_features = n_selected_features

    def fit(self, X, y, feature_names=None):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
              The training input samples
            y : numpy array, shape (n_samples, )
              The target values
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ----------
            None

            See Also
            --------

            examples
            --------

        """
        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        self.feature_names = dict(zip(features, feature_names))
        n_samples = X.shape[0]
        A_within = np.zeros((n_samples, n_samples))
        labels = np.unique(y)
        n_classes = labels.size
        for i in range(n_classes):
            sample_from_class = (y == labels[i])
            cross_samples = (sample_from_class[:, np.newaxis] & sample_from_class[np.newaxis, :])
            A_within[cross_samples] = 1.0 / np.count_nonzero(sample_from_class)
        L_within = np.eye(n_samples) - A_within
        L_between = np.ones((n_samples, n_samples)) / n_samples - A_within

        L_within = (L_within.T + L_within) / 2
        L_between = (L_between.T + L_between) / 2
        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)
        E = (E.T + E) / 2
        B = (B.T + B) / 2

        # we need only diagonal elements for trace calculation
        e = np.absolute(np.diag(E))
        b = np.absolute(np.diag(B))
        #b[b == 0] = 1e-14 # TODO: probably should be e[e == 0] = 1e-14?
        e[e == 0] = 1e-14
        features_indices = np.argsort(np.divide(b, e))[::-1][0:self.n_selected_features]
        lam = np.sum(b[features_indices]) / np.sum(e[features_indices])
        prev_lam = 0
        while (lam - prev_lam >= 1e-3):
            score = b - lam * e
            features_indices = np.argsort(score)[::-1][0:self.n_selected_features]
            prev_lam = lam
            lam = np.sum(b[features_indices]) / np.sum(e[features_indices])
        self.selected_features = features[features_indices]
        #return features_indices, score, lam

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
