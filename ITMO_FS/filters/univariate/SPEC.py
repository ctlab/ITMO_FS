from logging import getLogger

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import pairwise_distances

from ...utils import l21_norm, matrix_norm, power_neg_half, BaseTransformer


class SPEC(BaseTransformer):
    """Spectral Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    k : int
        Amount of clusters to find.
    gamma : callable
        An "increasing function that penalizes high frequency components".
        Default is gamma(x) = x^2.
    sigma : float
        Parameter for the weighting scheme.
    phi_type : int (1, 2 or 3)
        Type of feature ranking function to use.

    Notes
    -----
    For more details see `this paper <http://www.public.asu.edu/~huanliu/papers/icml07.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import SPEC
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> model = SPEC(3).fit(X, y)
    >>> model.selected_features_
    array([0, 1, 4], dtype=int64)
    >>> model = SPEC(3).fit(X)
    >>> model.selected_features_
    array([3, 4, 1], dtype=int64)
    """
    def __init__(self, n_features, k=2, gamma=(lambda x: x ** 2), sigma=0.5,
                 phi_type=3):
        self.n_features = n_features
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        self.phi_type = phi_type

    def __scheme(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))

    def __phi1(self, cosines, eigvals, k):
        return np.sum(cosines * cosines * self.gamma(eigvals))

    def __phi2(self, cosines, eigvals, k):
        return (np.sum(cosines[1:] * cosines[1:] * self.gamma(eigvals[1:]))
                / np.sum(cosines[1:] * cosines[1:]))

    def __phi3(self, cosines, eigvals, k):
        return np.sum(cosines[1:k] * cosines[1:k]
                      * (self.gamma(2) - self.gamma(eigvals[1:k])))

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The target values. If present, label values are used to
            construct the similarity graph and the amount of classes
            overrides k.

        Returns
        -------
        None
        """
        def calc_weight(f):
            f_norm = np.sqrt(D).dot(f)
            f_norm /= np.linalg.norm(f_norm)

            cosines = np.apply_along_axis(
                lambda vec: np.dot(vec / np.linalg.norm(vec), f_norm), 0,
                eigvectors)
            return phi(cosines, eigvals, k)

        if self.phi_type == 1:
            phi = self.__phi1
        elif self.phi_type == 2:
            phi = self.__phi2
        elif self.phi_type == 3:
            phi = self.__phi3
        else:
            getLogger(__name__).error(
                "phi_type should be 1, 2 or 3, %d passed", self.phi_type)
            raise ValueError(
                "phi_type should be 1, 2 or 3, %d passed" % self.phi_type)

        n_samples = X.shape[0]

        if y is None:
            if self.k > n_samples:
                getLogger(__name__).error(
                    "Cannot find %d clusters with n_samples = %d",
                    self.k, n_samples)
                raise ValueError(
                    "Cannot find %d clusters with n_samples = %d"
                    % (self.k, n_samples))
            k = self.k
            graph = np.ones((n_samples, n_samples))
            W = graph * pairwise_distances(
                X, metric=lambda x, y: self.__scheme(x, y))
        else:
            values, counts = np.unique(y, return_counts=True)
            values_dict = dict(zip(values, counts))
            k = len(values)
            W = pairwise_distances(
                y.reshape(-1, 1),
                metric=lambda x, y: (x[0] == y[0]) / values_dict[x[0]])

        getLogger(__name__).info("W: %s", W)

        D = np.diag(W.sum(axis=1))
        getLogger(__name__).info("D: %s", D)
        L = D - W
        getLogger(__name__).info("L: %s", L)
        L_norm = power_neg_half(D).dot(L).dot(power_neg_half(D))
        getLogger(__name__).info("Normalized L: %s", L_norm)
        eigvals, eigvectors = eigh(a=L_norm)
        getLogger(__name__).info(
            "Eigenvalues for normalized L: %s, eigenvectors: %s",
            eigvals, eigvectors)

        self.feature_scores_ = np.apply_along_axis(
            lambda f: calc_weight(f), 0, X)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)
        if self.phi_type == 3:
            ranking = ranking[::-1]
        self.selected_features_ = ranking[:self.n_features]
