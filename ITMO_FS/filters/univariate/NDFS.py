from logging import getLogger

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from ...utils import l21_norm, matrix_norm, power_neg_half, BaseTransformer


class NDFS(BaseTransformer):
    """Nonnegative Discriminative Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    c : int
        Amount of clusters to find.
    k : int
        Amount of nearest neighbors to use while building the graph.
    alpha : float
        Parameter in the objective function.
    beta : float
        Regularization parameter in the objective function.
    gamma : float
        Parameter in the objective function that controls the orthogonality
        condition.
    sigma : float
        Parameter for the weighting scheme.
    max_iterations : int
        Maximum amount of iterations to perform.
    epsilon : positive float
        Specifies the needed residual between the target functions from
        consecutive iterations. If the residual is smaller than epsilon, the
        algorithm is considered to have converged.

    See Also
    --------
    http://www.nlpr.ia.ac.cn/2012papers/gjhy/gh27.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import NDFS
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> model = NDFS(3).fit(X, y)
    >>> model.selected_features_
    array([0, 3, 4], dtype=int64)
    >>> model = NDFS(3).fit(X)
    >>> model.selected_features_
    array([3, 4, 1], dtype=int64)
    """
    def __init__(self, n_features, c=2, k=3, alpha=1, beta=1, gamma=10e8,
                 sigma=1, max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.c = c
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def __scheme(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (self.sigma ** 2))

    def _fit(self, X, y, **kwargs):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The target values or their one-hot encoding that are used to
            compute F. If not present, a k-means clusterization algorithm
            is used. If present, n_classes should be equal to c.

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
            n_neighbors=self.k,
            algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)
        getLogger(__name__).info("Nearest neighbors graph: %s", graph)

        S = graph * pairwise_distances(
            X, metric=lambda x, y: self.__scheme(x, y))
        getLogger(__name__).info("S: %s", S)
        A = np.diag(S.sum(axis=0))
        getLogger(__name__).info("A: %s", A)
        L = power_neg_half(A).dot(A - S).dot(power_neg_half(A))
        getLogger(__name__).info("L: %s", L)

        if y is not None:
            if len(y.shape) == 2:
                Y = y
            else:
                Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        else:
            if self.c > n_samples:
                getLogger(__name__).error(
                    "Cannot find %d clusters with n_samples = %d", self.c,
                    n_samples)
                raise ValueError(
                    "Cannot find %d clusters with n_samples = %d"
                    % (self.c, n_samples))
            Y = self.__run_kmeans(X)
        getLogger(__name__).info("Transformed Y: %s", Y)
        F = Y.dot(power_neg_half(Y.T.dot(Y)))
        getLogger(__name__).info("F: %s", F)
        D = np.eye(self.n_features_)
        In = np.eye(n_samples)
        Ic = np.eye(Y.shape[1])

        previous_target = -1
        for _ in range(self.max_iterations):
            M = (L + self.alpha
                * (In - X.dot(
                    np.linalg.inv(X.T.dot(X) + self.beta * D)).dot(X.T)))
            getLogger(__name__).info("M: %s", M)
            F = (F * ((self.gamma * F)
                       / (M.dot(F) + self.gamma * F.dot(F.T).dot(F))))
            getLogger(__name__).info("F: %s", F)
            W = np.linalg.inv(X.T.dot(X) + self.beta * D).dot(X.T.dot(F))
            getLogger(__name__).info("W: %s", W)
            diag = 2 * matrix_norm(W)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)
            getLogger(__name__).info("D: %s", D)

            target = (np.trace(F.T.dot(L).dot(F))
                + self.alpha * (np.linalg.norm(X.dot(W) - F) ** 2
                    + self.beta * l21_norm(W))
                + self.gamma * (np.linalg.norm(F.T.dot(F) - Ic) ** 2) / 2)
            getLogger(__name__).info("New target value: %d", target)
            if abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        getLogger(__name__).info("Ended up with W: %s", W)
        self.feature_scores_ = matrix_norm(W)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]

    def __run_kmeans(self, X):
        kmeans = KMeans(n_clusters=self.c, copy_x=True)
        kmeans.fit(X)
        labels = kmeans.labels_
        getLogger(__name__).info("Labels from KMeans: %s", labels)
        return OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
