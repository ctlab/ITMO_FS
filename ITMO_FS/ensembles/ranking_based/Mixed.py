import random

from ITMO_FS.filters.univariate.measures import select_k_best


class Mixed:
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
    from ITMO_FS.filters.univariate.measures import spearman_corr,pearson_corr
    from ITMO_FS.hybrid.Mixed import Mixed
    from sklearn.datasets import make_classification
    x, y = make_classification(1000, 50, n_informative = 5, n_redundant = 3, n_repeated = 2, shuffle = True)
    mixed = Mixed([spearman_corr, pearson_corr])
    print(mixed.run(x, y, 20))
    
    """

    __filters = []

    def __init__(self, filters):
        self.__filters = filters

    """
    Runs mixed hybrid method

    Parameters
    ----------
    X : array-like, shape (n_samples,n_features)
        The input samples.
    y : array-like, shape (n_samples)
        The classes for the samples.
    k : int
        The number of features to select.
    
    Returns
    ------
    array-like k selected features
    
    """

    def run(self, X, y, k):
        result = []
        filter_results = list(map(lambda fn: select_k_best(k)(dict(zip(list(range(X.shape[1])), fn(X, y)))),
                                  self.__filters))  # call every filter on input data, then select k best for each of
        # them

        place = 0
        while (len(result) < k) and (place < k):
            placed_features = list(map(list, zip(*filter_results)))[
                place]  # take only features at index=place in filter array
            random.shuffle(placed_features)
            [result.append(z) for z in list(set(placed_features)) if z not in result]
            place += 1
        return result[0:k]
