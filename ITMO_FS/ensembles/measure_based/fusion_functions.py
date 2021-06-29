from numpy import dot


def weight_fusion(filter_scores, weights):
    """Calculate the weighted score of each feature.

    Parameters
    ----------
    filter_scores : array-like, shape (n_filters, n_features)
        Scores for all filters.
    weights : array-like, shape (n_filters,)
        Filter weights.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
    """
    return filter_scores.T.dot(weights)
