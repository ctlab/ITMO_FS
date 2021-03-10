import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
from ...utils import BaseTransformer, generate_features


# TODO requests changes for MultivariateFilter to be used there
class TraceRatioFisher(BaseTransformer):
    """
        Creates TraceRatio(similarity based) feature selection filter
        performed in supervised way, i.e fisher version

        Parameters
        ----------
        n_features : int
            Number of features to select.

        Notes ----- For more details see `this paper
        <https://www.aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import TraceRatioFisher
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> tracer = TraceRatioFisher(3)
        >>> tracer.fit_transform(X, y)
        array([[1, 1, 2],
               [2, 2, 2],
               [3, 1, 3],
               [4, 3, 1],
               [5, 4, 4]])
    """

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def _fit(self, X, y):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
              The training input samples
            y : numpy array, shape (n_samples)
              The target values

            Returns
            ----------
            None
        """

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" % (
                    self.n_features, self.n_features_))

        features = generate_features(X)
        n_samples = X.shape[0]
        A_within = np.zeros((n_samples, n_samples))
        labels = np.unique(y)
        n_classes = labels.size
        for i in range(n_classes):
            sample_from_class = (y == labels[i])
            cross_samples = (
                sample_from_class[:, np.newaxis] & sample_from_class[
                    np.newaxis, :])
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
        # b[b == 0] = 1e-14 # TODO: probably should be e[e == 0] = 1e-14?
        e[e == 0] = 1e-14
        self.selected_features_ = np.argsort(np.divide(b, e))[::-1][
            0:self.n_features]
        lam = np.sum(b[self.selected_features_]) / np.sum(
            e[self.selected_features_])
        prev_lam = 0
        while lam - prev_lam >= 1e-3:
            score = b - lam * e
            self.selected_features_ = np.argsort(score)[::-1][
                0:self.n_features]
            prev_lam = lam
            lam = np.sum(b[self.selected_features_]) / np.sum(
                e[self.selected_features_])
