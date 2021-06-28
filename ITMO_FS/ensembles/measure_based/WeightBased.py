from logging import getLogger

import numpy as np
from sklearn.base import clone

from .fusion_functions import *
from ...utils import BaseTransformer, apply_cr, check_filters


class WeightBased(BaseTransformer):
    """Weight-based filter ensemble. The ensemble first computes all filter
    scores for the dataset and then aggregates them using a selected fusion
    function.

    Parameters
    ----------
    filters : collection
        Collection of filter objects. Filters should have a fit(X, y) method
        and a feature_scores_ field that contains scores for all features.
    cutting_rule : string or callable
        A cutting rule name defined in GLOB_CR or a callable with signature
        cutting_rule (features), which should return a list features ranked by
        some rule.
    fusion_function : callable
        A function with signature (filter_scores (array-like, shape
        (n_filters, n_features)), weights (array-like, shape (n_filters,)))
        that should return the aggregated weights for all features.
    weights : array-like
        An array of shape (n_filters,) defining the weights for input filters.

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.ensembles import WeightBased
    >>> from ITMO_FS.filters.univariate import UnivariateFilter
    >>> import numpy as np
    >>> filters = [UnivariateFilter('GiniIndex'),
    ... UnivariateFilter('FechnerCorr'),
    ... UnivariateFilter('SpearmanCorr'),
    ... UnivariateFilter('PearsonCorr')]
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> wb = WeightBased(filters, ("K best", 2)).fit(x, y)
    >>> wb.selected_features_
    array([4, 1], dtype=int64)
    """
    def __init__(self, filters, cutting_rule=("K best", 2),
                 fusion_function=weight_fusion, weights=None):
        self.filters = filters
        self.cutting_rule = cutting_rule
        self.fusion_function = fusion_function
        self.weights = weights

    def get_scores(self, X, y):
        """Return the normalized feature scores for all filters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        array-like, shape (n_filters, n_features) : feature scores
        """
        scores = np.vectorize(
            lambda f: clone(f).fit(X, y).feature_scores_,
            signature='()->(1)')(self.filters)
        getLogger(__name__).info("Scores for all filters: %s", scores)
        mins = np.min(scores, axis=1).reshape(-1, 1)
        maxs = np.max(scores, axis=1).reshape(-1, 1)
        return (scores - mins) / (maxs - mins)

    def __len__(self):
        """Return the number of filters used in the ensemble.

        Parameters
        ----------

        Returns
        -------
        int : number of filters
        """
        return len(self.filters)

    def _fit(self, X, y):
        """Fit the ensemble.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        check_filters(self.filters)
        getLogger(__name__).info(
            "Running WeightBased with filters: %s", self.filters)
        filter_scores = self.get_scores(X, y)
        getLogger(__name__).info(
            "Normalized scores for all filters: %s", filter_scores)
        if self.weights is None:
            weights = np.ones(len(self.filters)) / len(self.filters)
        else:
            weights = self.weights
        getLogger(__name__).info("Weights vector: %s", weights)
        self.feature_scores_ = self.fusion_function(filter_scores, weights)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        self.selected_features_ = apply_cr(self.cutting_rule)(
            self.feature_scores_)
