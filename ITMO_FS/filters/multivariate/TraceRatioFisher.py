import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *


# TODO X and y transformation for DataFrame support
# TODO requests changes for MultivariateFilter to be used there
class TraceRatioFisher(object):
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

    def run(self, X, y):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
              The training input samples
            y : numpy array, shape (n_samples, )
              The target values

            Returns
            ----------
            feature_indices : numpy array
                array of feature indices in X

            See Also
            --------

            examples
            --------

        """

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
        b[b == 0] = 1e-14
        features_indices = np.argsort(np.divide(b, e))[::-1][0:self.n_selected_features]
        lam = np.sum(b[features_indices]) / np.sum(e[features_indices])
        prev_lam = 0
        while (lam - prev_lam >= 1e-3):
            score = b - lam * e
            features_indices = np.argsort(score)[::-1][0:self.n_selected_features]
            prev_lam = lam
            lam = np.sum(b[features_indices]) / np.sum(e[features_indices])
        return features_indices, score, lam
