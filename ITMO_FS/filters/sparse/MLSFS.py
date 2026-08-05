import numpy as np
from sklearn.neighbors import NearestNeighbors

class MLSFS(object):
    """
        Performs the Semi-supervised sparse feature selection algorithm based on multi-view
Laplacian regularization.

        Parameters
        ----------
        p : int
            Number of features to select.
        labled : int
            Amount of first labled features.
        ds : list of int
            List of amount of features to each view.
        neighbor : int
            Amount of nearest neighbors to use while building the graph.
        mu : float, optional
            Parameter in the objective function.
        lam : float, optional
            Regularization parameter in the objective function.
        gamma : float, optional
            Parameter in the objective function that keeps view weights positive.
            Gamma should be greater than one.
        max_iterations : int, optional
            Maximum amount of iterations to perform.
        epsilon : positive float, optional
            Specifies the needed residual between the target functions from consecutive iterations. If the residual
            is smaller than epsilon, the algorithm is considered to have converged.

        See Also
        --------
        https://www.sciencedirect.com/science/article/abs/pii/S0262885615000748

        Examples
        --------

    """

    def __init__(self, p, labled, ds, neighbors, mu=1, lam=1, gamma=1.1, max_iterations=1000, epsilon=1e-5):
        self.p = p
        self.labled = labled
        self.ds = ds
        self.mu = mu
        self.lam = lam
        if gamma <= 1:
            raise ValueError("Gamma should be greater than one, %d passed" % gamma)
        self.gamma = gamma
        self.neighbors = neighbors
        self.max_iterations = max_iterations
        if epsilon < 0:
            raise ValueError("Epsilon should be positive, %d passed" % epsilon)
        self.epsilon = epsilon


    def run(self, X, y=None):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                The training input samples.
            y : numpy array, shape (n_samples) or (n_samples, n_classes), optional
                The target values of one or zero to first labled samples and zeros to other

            Returns
            ----------
            G : array-like, shape (n_features, c)
                Projection matrix.

            See Also
            --------

            Examples
            --------
            >>> from ITMO_FS.filters.sparse.MLSFS import MLSFS
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
            >>> y = np.array([1, 0, 1, 0, 0], dtype=np.integer)
            >>> model = MLSFS(p=3, labled=3, ds=[2, 3], neighbors=3)
            >>> weights = model.run(X)
            >>> model.feature_ranking(weights)
        """

        n_samples, n_features = X.shape
        n_views = len(self.ds)

        if len(y.shape) == 1:
            y = np.array([[y_el] for y_el in y])
        n_classes = y.shape[1]

        view_matrices = []
        cur_index = 0
        for i in range(n_views):
            matrix = []
            for x in X:
                matrix.append([x[j] for j in range(cur_index, cur_index + self.ds[i])])
            cur_index += self.ds[i]
            view_matrices.append(np.array(matrix).T)

        X = X.T

        S = [NearestNeighbors(n_neighbors=self.neighbors).fit(view.T).kneighbors_graph(view.T).toarray()
             for view in view_matrices]

        D = self.neighbors * np.eye(n_samples)
        Lv = [D - s for s in S]

        U = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            if i < self.labled:
                U[i, i] = 1e99
            else:
                U[i, i] = 1

        t = 0

        alphas = np.array([1 / n_views for i in range(n_views)])

        G = np.random.rand(n_features, n_classes)

        for iteration in range(self.max_iterations):
            L = np.sum(np.array([alphas[i] ** self.gamma * Lv[i] for i in range(n_views)]), 0)
            P = np.linalg.inv(L + U + self.mu * np.eye(n_samples))
            Q = U.dot(y) + self.mu * X.T.dot(G)
            F = P.dot(Q)
            A = X.dot(self.mu * np.eye(n_samples) - self.mu ** 2 * P.T).dot(X.T)
            B = self.mu * X.dot(P).dot(U).dot(y)
            W = np.zeros((n_features, n_features))
            for i in range(n_features):
                W[i, i] = (G[i].dot(G[i]) ** 1.5) / 4
            newG = np.linalg.inv(A + 4 * self.lam * W).dot(B)
            sum_for_alphas = sum([pow((1 / np.trace(F.T.dot(Lv[i]).dot(F))), (1 / (self.gamma - 1)))
                                  for i in range(n_views)])
            alphas = [pow((1 / np.trace(F.T.dot(Lv[i]).dot(F))), (1 / (self.gamma - 1))) / sum_for_alphas
                      for i in range(n_views)]

            diff = np.sum(np.abs(G - newG))
            print("diff ", diff)
            if (diff < self.epsilon):
                break
            G = newG

        return G

    def feature_ranking(self, G):
        """
            Choose p features.

            Parameters
            ----------
            G : array-like, shape (n_features, c)
                Feature weight matrix.

            p : amount of features to select

            Returns
            -------
            indices : array-like, shape(p)
                Indices of p selected features.
        """
        features_weights = np.sum(np.abs(G), 1)
        features_weights = [(i, features_weights[i]) for i in range(len(features_weights))]
        features_weights = sorted(features_weights, key=lambda el: el[1])
        return list(map(lambda el: el[0], features_weights[-self.p:]))

