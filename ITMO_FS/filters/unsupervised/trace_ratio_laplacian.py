from logging import getLogger

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ...utils import BaseTransformer

class TraceRatioLaplacian(BaseTransformer):
    """TraceRatio(similarity based) feature selection filter performed in
    unsupervised way, i.e laplacian version

    Parameters
    ----------
    n_features : int
        Amount of features to select.
    k : int
        Amount of nearest neighbors to use while building the graph.
    t : int
        constant for kernel function calculation
    epsilon : float
        Lambda change threshold.

    Notes
    -----
    For more details see `this paper <https://aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.unsupervised import TraceRatioLaplacian
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> tracer = TraceRatioLaplacian(2, k=2).fit(X)
    >>> tracer.selected_features_
    array([3, 1], dtype=int64)
    """
    def __init__(self, n_features, k=5, t=1, epsilon=1e-3):
        self.n_features = n_features
        self.k = k
        self.t = t
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-likey, shape (n_samples, n_features)
            The training input samples
        y : array-like, shape (n_samples,)
            The target values

        Returns
        -------
        None
        """
        n_samples = X.shape[0]

        if self.k >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.k, n_samples))

        graph = NearestNeighbors(
            n_neighbors=self.n_features,
            algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)
        getLogger(__name__).info("Nearest neighbors graph: %s", graph)

        A_within = graph * pairwise_distances(
            X, metric=lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / self.t))
        getLogger(__name__).info("A_within: %s", A_within)
        D_within = np.diag(A_within.sum(axis=1))
        getLogger(__name__).info("D_within: %s", D_within)
        L_within = D_within - A_within
        getLogger(__name__).info("L_within: %s", L_within)
        A_between = (D_within.dot(np.ones((n_samples, n_samples))).dot(D_within)
                     / np.sum(D_within))
        getLogger(__name__).info("A_between: %s", A_between)
        D_between = np.diag(A_between.sum(axis=1))
        getLogger(__name__).info("D_between: %s", D_between)
        L_between = D_between - A_between
        getLogger(__name__).info("L_between: %s", L_between)

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        getLogger(__name__).info("E: %s", e)
        getLogger(__name__).info("B: %s", b)
        lam = 0
        prev_lam = -1
        while lam - prev_lam >= self.epsilon:  # TODO: optimize
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
