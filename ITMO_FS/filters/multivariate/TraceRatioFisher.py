import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
from ...utils import BaseTransformer, generate_features

class TraceRatioFisher(BaseTransformer):
    """
        Creates TraceRatio(similarity based) feature selection filter
        performed in supervised way, i.e fisher version

        Parameters
        ----------
        n_features : int
            Number of features to select.
        epsilon : float
            Lambda change threshold.

        Notes
        -----
        For more details see `this paper
        <https://www.aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import TraceRatioFisher
        >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3], \
[3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
        >>> y = np.array([1, 2, 1, 1, 2])
        >>> tracer = TraceRatioFisher(3).fit(x, y)
        >>> tracer.selected_features_
        array([0, 1, 3], dtype=int64)
    """

    def __init__(self, n_features, epsilon=1e-3):
        self.n_features = n_features
        self.epsilon = epsilon

    def _fit(self, X, y):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
              The training input samples
            y : array-like, shape (n_samples)
              The target values

            Returns
            ----------
            None
        """

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" % (
                    self.n_features, self.n_features_))

        n_samples = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)
        counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}

        A_within = pairwise_distances(y.reshape(-1, 1), metric=lambda x, y:
            (x[0] == y[0]) / counts_d[x[0]])
        L_within = np.eye(n_samples) - A_within

        L_between = A_within - np.ones((n_samples, n_samples)) / n_samples

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        lam = 0
        prev_lam = -1
        while (lam - prev_lam >= self.epsilon):  # TODO: optimize
            score = b - lam * e
            self.selected_features_ = np.argsort(score)[::-1][
                0:self.n_features]
            prev_lam = lam
            lam = np.sum(b[self.selected_features_]) / np.sum(
                e[self.selected_features_])
        self.score_ = score
        self.lam_ = lam
