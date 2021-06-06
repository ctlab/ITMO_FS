from .fusion_functions import *
import numpy as np

from ...utils import BaseTransformer


class Mixed(BaseTransformer):
    """
        Performs feature selection based on several filters, selecting features
        this way:
                Get ranks from every filter from input.
                Then loops through, on every iteration=i
                    selects features on i position on every filter
                    then shuffles them, then adds to result list without
                    duplication,
                continues until specified number of features

        Parameters
        ----------
        filters: collection
            Collection of measure functions with signature measure(X, y) that
            should return an array of importance values for each feature.

        Examples
        --------
        >>> from ITMO_FS.filters.univariate.measures import *
        >>> from ITMO_FS.ensembles.ranking_based.Mixed import Mixed
        >>> import numpy as np
        >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
[3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
        >>> y = np.array([1, 3, 2, 1, 2])
        >>> mixed = Mixed([gini_index, chi2_measure], 2).fit(x, y)
        >>> mixed.selected_features_
        array([2, 4], dtype=int64)
    """

    def __init__(self, filters, k, fusion_function=best_goes_first_fusion):
        self.filters = filters
        self.k = k
        self.fusion_function = fusion_function

    def _fit(self, X, y):
        """
            Fits the ensemble.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

            Returns
            ------
            None
        """
        if self.k > self.n_features_:
            raise ValueError(
                "Cannot select %d best features with n_features = %d" %
                (self.k, self.n_features_))
        #TODO: some measures are 'lower is better', a simple argsort would not
        #work there - need to call a different ranking function
        self.filter_ranks_ = np.vectorize(lambda f: np.argsort(f(X, y))[::-1],
            signature='()->(1)')(self.filters)
        self.selected_features_ = self.fusion_function(self.filter_ranks_,
            self.k)
