from logging import getLogger

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from ...utils import knn_from_class, BaseTransformer


class STIR(BaseTransformer):
    """Feature selection using STIR algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    metric : str or callable
        Distance metric to use in kNN. If str, should be one of the standard
        distance metrics (e.g. 'euclidean' or 'manhattan'). If callable, should
        have the signature metric(x1 (array-like, shape (n,)), x2 (array-like,
        shape (n,))) that should return the distance between two vectors.
    k : int
        Number of constant nearest hits/misses.

    Notes
    -----
    For more details see `this paper <https://academic.oup.com/bioinformatics/article/35/8/1358/5100883>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import STIR
    >>> import numpy as np
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> model = STIR(2).fit(X, y)
    >>> model.selected_features_
    array([2, 0], dtype=int64)
    """
    def __init__(self, n_features, metric='manhattan', k=1):
        self.n_features = n_features
        self.metric = metric
        self.k = k

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The classes for the samples.

        Returns
        -------
        None
        """
        n_samples = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)

        if np.any(counts <= self.k):
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors because one of the classes "
                "has less than %d samples", self.k, self.k + 1)
            raise ValueError(
                "Cannot select %d nearest neighbors because one of the classes "
                "has less than %d samples" % (self.k, self.k + 1))

        x_normalized = MinMaxScaler().fit_transform(X)
        dm = pairwise_distances(x_normalized, x_normalized, self.metric)
        getLogger(__name__).info("Distance matrix: %s", dm)

        indices = np.arange(n_samples)
        hits_diffs = np.abs(
            np.vectorize(
                lambda index: (
                    x_normalized[index]
                    - x_normalized[knn_from_class(
                        dm, y, index, self.k, y[index])]),
                signature='()->(n,m)')(indices))
        getLogger(__name__).info("Hit differences matrix: %s", hits_diffs)
        misses_diffs = np.abs(
            np.vectorize(
                lambda index: (
                    x_normalized[index]
                    - x_normalized[knn_from_class(
                        dm, y, index, self.k, y[index], anyOtherClass=True)]),
                signature='()->(n,m)')(indices))
        getLogger(__name__).info("Miss differences matrix: %s", misses_diffs)

        H = np.mean(hits_diffs, axis=(0,1))
        getLogger(__name__).info("H: %s", H)
        M = np.mean(misses_diffs, axis=(0,1))
        getLogger(__name__).info("M: %s", M)
        var_H = np.var(hits_diffs, axis=(0,1))
        var_M = np.var(misses_diffs, axis=(0,1))

        # the 1 / (1 / |M| + 1 / |H|) ^ (1/2) multiplier is constant, we omit it
        self.feature_scores_ = (
            (M - H) * np.sqrt(2 * self.k * n_samples - 2)
            / (np.sqrt((self.k * n_samples - 1) * (var_H + var_M)) + 1e-15))
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        self.selected_features_ = np.argsort(self.feature_scores_)[::-1][
            :self.n_features]

