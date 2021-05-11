import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
from sklearn.neighbors import NearestNeighbors
from ...utils import BaseTransformer

# TODO requests changes for MultivariateFilter to be used there
class TraceRatioLaplacian(BaseTransformer):
    """
        Creates TraceRatio(similarity based) feature selection filter
        performed in unsupervised way, i.e laplacian version

        Parameters
        ----------
        n_features : int
            Amount of features to filter
        k : int
            number of neighbours to use for knn
        t : int
            constant for kernel function calculation
        epsilon : float
        	Lambda change threshold.

            - Note: in laplacian case only. In fisher it uses label similarity, i.e if both samples belong to same class

        Notes
        -----
        For more details see `this paper <https://aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.unsupervised.trace_ratio_laplacian import TraceRatioLaplacian
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[1, 1, 3, 1, 4],[2, 4, 3, 1, 5]])
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
            None
        """

        n_samples = X.shape[0]

        if self.k >= n_samples:
            raise ValueError("Cannot select %d nearest neighbors with n_samples = %d" % (self.k, n_samples))

        graph = NearestNeighbors(n_neighbors=self.n_features, algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)
        A_within = graph * pairwise_distances(X, metric=lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / self.t))
        D_within = np.diag(A_within.sum(axis=1))
        L_within = D_within - A_within
        A_between = D_within.dot(np.ones((n_samples, n_samples))).dot(D_within) / np.sum(D_within)
        D_between = np.diag(A_between.sum(axis=1))
        L_between = D_between - A_between

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        lam = 0
        prev_lam = -1
        while (lam - prev_lam >= self.epsilon):  # TODO: optimize
            score = b - lam * e
            self.selected_features_ = np.argsort(score)[::-1][0:self.n_features]
            prev_lam = lam
            lam = np.sum(b[self.selected_features_]) / np.sum(e[self.selected_features_])
        self.score_ = score
        self.lam_ = lam
