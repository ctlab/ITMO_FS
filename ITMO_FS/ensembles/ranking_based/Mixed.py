from logging import getLogger

import numpy as np

from .fusion_functions import *
from ...utils import BaseTransformer


class Mixed(BaseTransformer):
    """Perform feature selection based on several filters, selecting features
    this way:
        Get ranks from every filter from input.
        Then loops through, on every iteration=i
            selects features on i position on every filter
            then shuffles them, then adds to result list without
            duplication,
        continues until specified number of features

    Parameters
    ----------
    filters : collection
        Collection of measure functions with signature measure(X, y) that
        should return an array of importance values for each feature.
    n_features : int
        Amount of features to select.
    fusion_function : callable
        A function with signature (filter_ranks (array-like, shape
        (n_filters, n_features), k (int)) that should return the indices of k
        selected features based on the filter rankings.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate.measures import *
    >>> from ITMO_FS.ensembles.ranking_based.Mixed import Mixed
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> mixed = Mixed([gini_index, chi2_measure], 2).fit(x, y)
    >>> mixed.selected_features_
    array([2, 4], dtype=int64)
    """
    def __init__(self, filters, n_features,
                 fusion_function=best_goes_first_fusion):
        self.filters = filters
        self.n_features = n_features
        self.fusion_function = fusion_function

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
        #TODO: some measures are 'lower is better', a simple argsort would not
        #work there - need to call a different ranking function
        self.filter_ranks_ = np.vectorize(
            lambda f: np.argsort(f(X, y))[::-1],
            signature='()->(1)')(self.filters)
        getLogger(__name__).info("Filter ranks: %s", self.filter_ranks_)
        self.selected_features_ = self.fusion_function(
            self.filter_ranks_, self.n_features)
