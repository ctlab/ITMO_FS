import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances

from ...utils import l21_norm, matrix_norm, power_neg_half, BaseTransformer


class NDFS(BaseTransformer):
    """
        Performs the Nonnegative Discriminative Feature Selection algorithm.

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
            consecutive iterations. If the residual is smaller than epsilon,
            the algorithm is considered to have converged.

        See Also
        --------
        http://www.nlpr.ia.ac.cn/2012papers/gjhy/gh27.pdf

        Examples
        --------
        >>> from ITMO_FS.filters.univariate import NDFS
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[1, 1, 3, 1, 4],[2, 4, 3, 1, 5]])
        >>> y = np.array([1, 2, 1, 1, 2])
        >>> model = NDFS(3).fit(X, y)
        >>> model.selected_features_
        array([0, 2, 1], dtype=int64)
        >>> model = NDFS(3).fit(X)
        >>> model.selected_features_
        array([3, 4, 2], dtype=int64)
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
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                The training input samples.
            y : numpy array, shape (n_samples) or (n_samples, n_classes)
                The target values or their one-hot encoding that are used to
                compute F. If not present, a k-means clusterization algorithm
                is used. If present, n_classes should be equal to c.

            Returns
            ----------
            None
        """

        if self.epsilon < 0:
            raise ValueError(
                "Epsilon should be positive, %d passed" % self.epsilon)

        n_samples = X.shape[0]

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" % (
                self.n_features, self.n_features_))

        if self.c > n_samples:
            raise ValueError("Cannot find %d clusters with n_samples = %d" % (
            self.c, n_samples))

        if self.k >= n_samples:
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d" % (
                self.k, n_samples))

        graph = NearestNeighbors(n_neighbors=self.n_features, 
                                 algorithm='ball_tree').fit(
            X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)

        S = graph * pairwise_distances(X, metric=lambda x, y: self.__scheme(x, y))
        A = np.diag(S.sum(axis=0))
        L = power_neg_half(A).dot(A - S).dot(power_neg_half(A))

        if y is not None:
            if len(y.shape) == 2:
                Y = y
            else:
                Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        else:
            Y = self.__run_kmeans(X)
        F = Y.dot(power_neg_half(Y.T.dot(Y)))
        D = np.eye(self.n_features_)
        I = np.eye(n_samples)

        previous_target = 0
        for step in range(self.max_iterations):
            M = L + self.alpha * (I - X.dot(
                np.linalg.inv(X.T.dot(X) + self.beta * D)).dot(X.T))
            F = F * ((self.gamma * F) / (
                        M.dot(F) + self.gamma * F.dot(F.T).dot(F)))
            W = np.linalg.inv(X.T.dot(X) + self.beta * D).dot(X.T.dot(F))
            diag = 2 * matrix_norm(W)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)

            target = np.trace(F.T.dot(L).dot(F)) + self.alpha * (
                        np.linalg.norm(X.dot(W) - F) + self.beta * l21_norm(W))
            if step > 0 and abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        ndfs_score = matrix_norm(W)
        ranking = np.argsort(ndfs_score)[::-1]
        self.selected_features_ = ranking[:self.n_features]

    def __run_kmeans(self, X):
        kmeans = KMeans(n_clusters=self.c, copy_x=True)
        kmeans.fit(X)
        labels = kmeans.labels_
        return OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
