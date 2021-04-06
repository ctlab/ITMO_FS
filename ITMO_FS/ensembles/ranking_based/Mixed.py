from .fusion_functions import *

from ...utils import BaseTransformer

class Mixed(BaseTransformer):
    """
        Performs feature selection based on several filters, selecting features this way:
        Get ranks from every filter from input. Then loops through, on every iteration=i 
        selects features on i position on every filter then shuffles them, then adds to result list
        without duplication, continues until specified number of features
        
        Parameters
        ----------
        filters: list of filter functions

        Examples
        --------
        >>> from ITMO_FS.filters.univariate.measures import spearman_corr,pearson_corr
        >>> from ITMO_FS.ensembles.ranking_based.Mixed import Mixed
        >>> from sklearn.datasets import make_classification
        >>> x, y = make_classification(1000, 50, n_informative = 5, n_redundant = 3, n_repeated = 2, shuffle = True)
        >>> mixed = Mixed([spearman_corr, pearson_corr])
        >>> mixed.fit()
        >>> mixed.transform(x, 20).shape
        (1000, 20)
    """

    def __init__(self, filters, k, fusion_function=best_goes_first_fusion):
        self.filters = filters
        self.k = k
        self.fusion_function = fusion_function

    def _fit(self, X, y):
        if self.k > self.n_features_:
            raise ValueError("Cannot select %d best features with n_features = %d" % (self.k, self.n_features_))
        self.filter_results_ = list(
            map(lambda fn: sorted(dict(enumerate(fn(X, y))).items(), key=lambda kv: kv[1], reverse=True),
                self.filters))  # call every filter on input data, then select k best for each of them
        self.selected_features_ = self.fusion_function(self.filter_results_, self.k)
