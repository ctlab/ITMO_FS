from logging import getLogger

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

from ...utils import l21_norm, matrix_norm, BaseTransformer


class UDFS(BaseTransformer):
    """Unsupervised Discriminative Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    c : int
        Amount of clusters to find.
    k : int
        Amount of nearest neighbors to use while building the graph.
    gamma : float
        Regularization term in the target function.
    l : float
        Parameter that controls the invertibility of the matrix used in
        computing of B.
    max_iterations : int
        Maximum amount of iterations to perform.
    epsilon : positive float
        Specifies the needed residual between the target functions from
        consecutive iterations. If the residual is smaller than epsilon, the
        algorithm is considered to have converged.

    Notes
    -----
    For more details see `this paper <https://www.ijcai.org/Proceedings/11/Papers/267.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.unsupervised import UDFS
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> dataset = make_classification(n_samples=500, n_features=100,
    ... n_informative=5, n_redundant=0, random_state=42, shuffle=False,
    ... n_clusters_per_class=1)
    >>> X, y = np.array(dataset[0]), np.array(dataset[1])
    >>> model = UDFS(5).fit(X)
    >>> model.selected_features_
    array([ 2,  3, 19, 90, 92], dtype=int64)
    """
    def __init__(self, n_features, c=2, k=3, gamma=1, l=1e-6,
                 max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.c = c
        self.k = k
        self.gamma = gamma
        self.l = l
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like
            The target values (ignored).

        Returns
        -------
        None
        """
        def construct_S(arr):
            S = np.zeros((n_samples, self.k + 1))
            for idx in range(self.k + 1):
                S[arr[idx], idx] = 1
            return S

        n_samples = X.shape[0]

        if self.c > n_samples:
            getLogger(__name__).error(
                "Cannot find %d clusters with n_samples = %d",
                self.c, n_samples)
            raise ValueError(
                "Cannot find %d clusters with n_samples = %d"
                % (self.c, n_samples))

        if self.k >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.k, n_samples))

        indices = list(range(n_samples))
        I = np.eye(self.k + 1)
        H = I - np.ones((self.k + 1, self.k + 1)) / (self.k + 1)

        neighbors = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm='ball_tree').fit(X).kneighbors(X, return_distance=False)
        getLogger(__name__).info("Neighbors graph: %s", neighbors)
        X_centered = np.apply_along_axis(
            lambda arr: X[arr].T.dot(H), 1, neighbors)

        S = np.apply_along_axis(lambda arr: construct_S(arr), 1, neighbors)
        getLogger(__name__).info("S: %s", S)
        B = np.vectorize(
            lambda idx: np.linalg.inv(X_centered[idx].T.dot(X_centered[idx])
                        + self.l * I),
            signature='()->(1,1)')(indices)
        getLogger(__name__).info("B: %s", B)
        Mi = np.vectorize(
            lambda idx: S[idx].dot(H).dot(B[idx]).dot(H).dot(S[idx].T),
            signature='()->(1,1)')(indices)
        M = X.T.dot(Mi.sum(axis=0)).dot(X)
        getLogger(__name__).info("M: %s", M)

        D = np.eye(self.n_features_)
        previous_target = -1
        for step in range(self.max_iterations):
            P = M + self.gamma * D
            getLogger(__name__).info("P: %s", P)
            _, W = eigh(a=P, subset_by_index=[0, self.c - 1])
            getLogger(__name__).info("W: %s", W)
            diag = 2 * matrix_norm(W)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)
            getLogger(__name__).info("D: %s", D)

            target = np.trace(W.T.dot(M).dot(W)) + self.gamma * l21_norm(W)
            getLogger(__name__).info("New target value: %d", target)
            if abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        getLogger(__name__).info("Ended up with W = %s", W)
        self.feature_scores_ = matrix_norm(W)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]
