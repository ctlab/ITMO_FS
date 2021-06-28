from logging import getLogger

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from ...utils import BaseTransformer, generate_features

class TraceRatioFisher(BaseTransformer):
    """Creates TraceRatio(similarity based) feature selection filter
    performed in supervised way, i.e. fisher version

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
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> tracer = TraceRatioFisher(3).fit(x, y)
    >>> tracer.selected_features_
    array([0, 1, 3], dtype=int64)
    """
    def __init__(self, n_features, epsilon=1e-3):
        self.n_features = n_features
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples
        y : array-like, shape (n_samples,)
            The target values

        Returns
        -------
        None
        """
        n_samples = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)
        counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}
        getLogger(__name__).info("Class counts: %s", counts_d)

        A_within = pairwise_distances(
            y.reshape(-1, 1), metric=lambda x, y: (
                (x[0] == y[0]) / counts_d[x[0]]))
        L_within = np.eye(n_samples) - A_within
        getLogger(__name__).info("A_w: %s", A_within)
        getLogger(__name__).info("L_w: %s", L_within)

        L_between = A_within - np.ones((n_samples, n_samples)) / n_samples
        getLogger(__name__).info("L_b: %s", L_between)

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        getLogger(__name__).info("E: %s", e)
        getLogger(__name__).info("B: %s", b)
        lam = 0
        prev_lam = -1
        while (lam - prev_lam >= self.epsilon):  # TODO: optimize
            score = b - lam * e
            getLogger(__name__).info("Score: %s", score)
            self.selected_features_ = np.argsort(score)[::-1][:self.n_features]
            getLogger(__name__).info(
                "New selected set: %s", self.selected_features_)
            prev_lam = lam
            lam = (np.sum(b[self.selected_features_])
                   / np.sum(e[self.selected_features_]))
            getLogger(__name__).info("New lambda: %d", lam)
        self.score_ = score
        self.lam_ = lam
