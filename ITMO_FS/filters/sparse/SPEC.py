import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from ...utils import knn, l21_norm, matrix_norm, power_neg_half


class SPEC(object):
    """
        Performs the Spectral Feature Selection algorithm.

        Parameters
        ----------
        p : int
            Number of features to select.
        k : int, optional
            Amount of clusters to find.
        gamma : callable, optional
            An "increasing function that penalizes high frequency components". Default is gamma(x) = x^2.
        sigma : float, optional
            Parameter for the weighting scheme.
        phi_type : int (1, 2 or 3), optional
            Type of feature ranking function to use.

        Notes
        -----
        For more details see `this paper <https://www.ijcai.org/Proceedings/11/Papers/267.pdf/>`_.


        Examples
        --------

    """

    def __init__(self, p, k=5, gamma=(lambda x: x ** 2), sigma=0.5, phi_type=1):
        self.p = p
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        if phi_type == 1:
            self.phi = self.__phi1
        elif phi_type == 2:
            self.phi = self.__phi2
        elif phi_type == 3:
            self.phi = self.__phi3
        else:
            raise ValueError("phi_type should be 1, 2 or 3, %d passed" % phi_type)

    def __scheme(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (4 * self.sigma ** 2))

    def __phi1(self, cosines, eigvals):
        return np.sum(cosines * cosines * self.gamma(eigvals))

    def __phi2(self, cosines, eigvals):
        return np.sum(cosines[1:] * cosines[1:] * self.gamma(eigvals[1:])) / np.sum(cosines[1:] * cosines[1:])

    def __phi3(self, cosines, eigvals):
        return np.sum(cosines * cosines * (self.gamma(2) - self.gamma(eigvals)))

    def run(self, X, y=None):
        """
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                The training input samples.
            y : numpy array, optional
                The target values. If present, label values are used to construct the similarity graph and the amount of classes overrides k.

            Returns
            ----------
            W : array-like, shape (n_features)
                Feature weight matrix.

            See Also
            --------

            Examples
            --------
            >>> from ITMO_FS.filters.sparse import SPEC
            >>> from sklearn.datasets import make_classification
            >>> import numpy as np
            >>> dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
            >>> data, target = np.array(dataset[0]), np.array(dataset[1])
            >>> model = SPEC(p=5, k=2)
            >>> weights = model.run(data, target)
            >>> print(model.feature_ranking(weights))
        """

        def calc_weight(f):
            f_norm = np.sqrt(D).dot(f)
            f_norm /= np.linalg.norm(f_norm)

            cosines = np.apply_along_axis(lambda vec: 1 - cosine(f_norm, vec), 0, eigvectors)
            return self.phi(cosines, eigvals)

        n_samples, n_features = X.shape
        if (y == None).any():
            graph = np.ones((n_samples, n_samples))
            indices = [[(i, j) for j in range(n_samples)] for i in range(n_samples)]
            func = np.vectorize(lambda xy: graph[xy[0]][xy[1]] * self.__scheme(X[xy[0]], X[xy[1]]), signature='(1)->()')
            W = func(indices)
        else:
            values, counts = np.unique(y, return_counts=True)
            values_dict = dict(zip(values, counts))
            self.k = len(values)
            W = np.array(
                [[(lambda i, j: 1 / values_dict[y[i]] if y[i] == y[j] else 0)(i, j) for j in range(n_samples)] for i in
                 range(n_samples)])

        D = np.diag(W.sum(axis=1))
        L = D - W
        L_norm = power_neg_half(D).dot(L).dot(power_neg_half(D))
        eigvals, eigvectors = eigh(a=L_norm)

        return np.apply_along_axis(lambda f: calc_weight(f), 0, X)

    def feature_ranking(self, W):
        """
            Calculate the SPEC score for a feature weight matrix.

            Parameters
            ----------
            W : array-like, shape (n_features)
                Feature weight matrix.

            Returns
            -------
            indices : array-like, shape(p)
                Indices of p selected features.
        """
        ranking = np.argsort(W)
        if self.phi == self.__phi3:
            ranking = ranking[::-1]
        return ranking[:self.p]
