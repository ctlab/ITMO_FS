from functools import partial, update_wrapper
from math import exp

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from ...utils.information_theory import conditional_entropy
from ...utils.information_theory import entropy
from ...utils.qpfs_body import qpfs_body
from ...utils.functions import knn_from_class


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def fit_criterion_measure(x, y):
    """Calculate the FitCriterion score for features. Bigger values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://core.ac.uk/download/pdf/191234514.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fit_criterion_measure
    >>> import numpy as np
    >>> x = np.array([[1, 2, 4, 1, 1], [2, 2, 2, 1, 2], [3, 5, 1, 1, 4],
    ... [1, 1, 1, 1, 4], [2, 2, 2, 1, 5]])
    >>> y = np.array([1, 2, 3, 1, 2])
    >>> fit_criterion_measure(x, y)
    array([1. , 0.8, 0.8, 0.4, 0.6])
    """
    def count_hits(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        means = {cl: np.mean(splits[cl]) for cl in classes}
        devs = {cl: np.var(splits[cl]) for cl in classes}
        distances = np.vectorize(
            lambda x_val: {cl: (
                abs(x_val - means[cl])
                / (devs[cl] + 1e-10)) for cl in classes})(feature)
        return np.sum(np.vectorize(lambda d: min(d, key=d.get))(distances) == y)

    classes = np.unique(y)
    return np.apply_along_axis(count_hits, 0, x) / x.shape[0]


def f_ratio_measure(x, y):
    """Calculate Fisher score for features. Bigger values mean more important
    features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import f_ratio_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> f_ratio_measure(x, y)
    array([0.6 , 0.2 , 1.  , 0.12, 5.4 ])
    """
    def __F_ratio(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        mean_feature = np.mean(feature)
        inter_class = np.sum(
            np.vectorize(lambda cl: (
                counts_d[cl]
                * np.power(mean_feature - np.mean(splits[cl]), 2)))(classes))
        intra_class = np.sum(
            np.vectorize(lambda cl: (
                counts_d[cl]
                * np.var(splits[cl])))(classes))
        return inter_class / (intra_class + 1e-10)

    classes, counts = np.unique(y, return_counts=True)
    counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}
    return np.apply_along_axis(__F_ratio, 0, x)


def gini_index(x, y):
    """Calculate Gini index for features. Bigger values mean more important
    features. This measure works best with discrete features due to being based
    on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import gini_index
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> gini_index(x, y)
    array([0.14      , 0.04      , 0.64      , 0.24      , 0.37333333])
    """
    def __gini(feature):
        values, counts = np.unique(feature, return_counts=True)
        counts_d = {val: counts[idx] for idx, val in enumerate(values)}
        total_sum = np.sum(
            np.vectorize(
                lambda val: (
                    np.sum(
                        np.square(
                            np.unique(
                                y[feature == val], return_counts=True)[1]))
                    / counts_d[val]))(values))
        return total_sum / x.shape[0] - prior_prob_squared_sum

    classes, counts = np.unique(y, return_counts=True)
    prior_prob_squared_sum = np.sum(np.square(counts / x.shape[0]))

    return np.apply_along_axis(__gini, 0, x)


def su_measure(x, y):
    """SU is a correlation measure between the features and the class
    calculated via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y)). Bigger values
    mean more important features. This measure works best with discrete
    features due to being based on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import su_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> su_measure(x, y)
    array([0.28694182, 0.13715115, 0.79187567, 0.47435099, 0.67126949])
    """
    def __SU(feature):
        entropy_x = entropy(feature)
        return (2 * (entropy_x - conditional_entropy(y, feature))
                  / (entropy_x + entropy_y))

    entropy_y = entropy(y)
    return np.apply_along_axis(__SU, 0, x)

# TODO CONCORDATION COEF

def kendall_corr(x, y):
    """Calculate Sample sign correlation (Kendall correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import kendall_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> kendall_corr(x, y)
    array([-0.1,  0.2, -0.4, -0.2,  0.2])
    """
    def __kendall_corr(feature):
        k_corr = 0.0
        for i in range(len(feature)):
            k_corr += np.sum(np.sign(feature[i] - feature[i + 1:])
                             * np.sign(y[i] - y[i + 1:]))
        return 2 * k_corr / (feature.shape[0] * (feature.shape[0] - 1))

    return np.apply_along_axis(__kendall_corr, 0, x)


def fechner_corr(x, y):
    """Calculate Sample sign correlation (Fechner correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fechner_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> fechner_corr(x, y)
    array([-0.2,  0.2, -0.4, -0.2, -0.2])
    """
    y_dev = y - np.mean(y)
    x_dev = x - np.mean(x, axis=0)
    return np.sum(np.sign(x_dev.T * y_dev), axis=1) / x.shape[0]

def reliefF_measure(x, y, k_neighbors=1):
    """Calculate ReliefF measure for each feature. Bigger values mean more
    important features.

    Note:
    Only for complete x
    Rather than repeating the algorithm m(TODO Ask Nikita about user defined)
    times, implement it exhaustively (i.e. n times, once for each instance)
    for relatively small n (up to one thousand).

    Calculates spearman correlation for each feature.
    Spearman's correlation assesses monotonic relationships (whether linear or
    not). If there are no repeated data values, a perfect Spearman correlation
    of +1 or −1 occurs when each of the variables is a perfect monotone
    function of the other.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    k_neighbors : int, optional
        The number of neighbors to consider when assigning feature importance
        scores. More neighbors results in more accurate scores but takes
        longer. Selection of k hits and misses is the basic difference to
        Relief and ensures greater robustness of the algorithm concerning noise.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and
    review. Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import reliefF_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1], [1, 2, 1, 4, 2], [4, 3, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2, 1, 2])
    >>> reliefF_measure(x, y)
    array([-0.14285714, -0.57142857,  0.10714286, -0.14285714,  0.07142857])
    >>> reliefF_measure(x, y, k_neighbors=2)
    array([-0.07142857, -0.17857143, -0.07142857, -0.0952381 , -0.17857143])
    """
    def __calc_misses(index):
        misses_diffs_classes = np.abs(
            np.vectorize(
                lambda cl: (
                        x[index]
                        - x[knn_from_class(dm, y, index, k_neighbors, cl)])
                    * prior_prob[cl],
                signature='()->(n,m)')(classes[classes != y[index]]))
        return (np.sum(np.sum(misses_diffs_classes, axis=1), axis=0)
            / (1 - prior_prob[y[index]]))

    classes, counts = np.unique(y, return_counts=True)
    if np.any(counts <= k_neighbors):
        raise ValueError(
            "Cannot calculate relieff measure because one of theclasses has "
            "less than %d samples" % (k_neighbors + 1))
    prior_prob = dict(zip(classes, np.array(counts) / len(y)))
    n_samples = x.shape[0]
    n_features = x.shape[1]
    # use manhattan distance instead of euclidean
    dm = pairwise_distances(x, x, 'manhattan')

    indices = np.arange(n_samples)
    # use abs instead of square because of manhattan distance
    hits_diffs = np.abs(
        np.vectorize(
            lambda index: (
                x[index]
                - x[knn_from_class(dm, y, index, k_neighbors, y[index])]),
            signature='()->(n,m)')(indices))
    H = np.sum(hits_diffs, axis=(0,1))

    misses_sum_diffs = np.vectorize(
        lambda index: __calc_misses(index),
        signature='()->(n)')(indices)
    M = np.sum(misses_sum_diffs, axis=0)

    weights = M - H
    # dividing by m * k guarantees that all final weights
    # will be normalized within the interval [ − 1, 1].
    weights /= n_samples * k_neighbors
    # The maximum and minimum values of A are determined over the entire
    # set of instances.
    # This normalization ensures that weight updates fall
    # between 0 and 1 for both discrete and continuous features.
    with np.errstate(divide='ignore', invalid="ignore"):  # todo
        return weights / (np.amax(x, axis=0) - np.amin(x, axis=0))


def relief_measure(x, y, m=None, random_state=42):
    """Calculate Relief measure for each feature. This measure is supposed to
    work only with binary classification datasets; for multi-class problems use
    the ReliefF measure. Bigger values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    m : int, optional
        Amount of iterations to do. If not specified, n_samples iterations
        would be performed.
    random_state : int, optional
        Random state for numpy random.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and
    review. Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import relief_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> relief_measure(x, y)
    array([ 0.    , -0.6   , -0.1875, -0.15  , -0.4   ])
    """
    weights = np.zeros(x.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 1:
        raise ValueError("Cannot calculate relief measure with 1 class")
    if 1 in counts:
        raise ValueError(
            "Cannot calculate relief measure because one of the classes has "
            "only 1 sample")

    n_samples = x.shape[0]
    n_features = x.shape[1]
    if m is None:
        m = n_samples

    x_normalized = MinMaxScaler().fit_transform(x)
    dm = euclidean_distances(x_normalized, x_normalized)
    indices = np.random.default_rng(random_state).integers(
        low=0, high=n_samples, size=m)
    objects = x_normalized[indices]
    hits_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(dm, y, index, 1, y[index])]),
            signature='()->(n,m)')(indices))
    misses_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(
                    dm, y, index, 1, y[index], anyOtherClass=True)]),
            signature='()->(n,m)')(indices))

    H = np.sum(hits_diffs, axis=(0,1))
    M = np.sum(misses_diffs, axis=(0,1))

    weights = M - H

    return weights / m


def chi2_measure(x, y):
    """Calculate the Chi-squared measure for each feature. Bigger values mean
    more important features. This measure works best with discrete features due
    to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Example
    -------
    >>> from ITMO_FS.filters.univariate import chi2_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> chi2_measure(x, y)
    array([ 1.875     ,  0.83333333, 10.        ,  3.75      ,  6.66666667])
    """
    def __chi2(feature):
        values, counts = np.unique(feature, return_counts=True)
        values_map = {val: idx for idx, val in enumerate(values)}
        splits = {cl: np.array([values_map[val] for val in feature[y == cl]]) 
            for cl in classes}
        e = np.vectorize(
            lambda cl: prior_probs[cl] * counts,
            signature='()->(1)')(classes)
        n = np.vectorize(
            lambda cl: np.bincount(splits[cl], minlength=values.shape[0]),
            signature='()->(1)')(classes)
        return np.sum(np.square(e - n) / e)

    classes, counts = np.unique(y, return_counts=True)
    prior_probs = {cl: counts[idx] / x.shape[0] for idx, cl
        in enumerate(classes)}
    
    return np.apply_along_axis(__chi2, 0, x)


#
# def __contingency_matrix(labels_true, labels_pred):
#     """Build a contingency matrix describing the relationship between labels.
#         Parameters
#         ----------
#         labels_true : int array, shape = [n_samples]
#             Ground truth class labels to be used as a reference
#         labels_pred : array, shape = [n_samples]
#             Cluster labels to evaluate
#         Returns
#         -------
#         contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
#             Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
#             true class :math:`i` and in predicted class :math:`j`. If
#             ``eps is None``, the dtype of this array will be integer. If ``eps`` is
#             given, the dtype will be float.
#         """
#     classes, class_idx = np.unique(labels_true, return_inverse=True)
#     clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
#     n_classes = classes.shape[0]
#     n_clusters = clusters.shape[0]
#     # Using coo_matrix to accelerate simple histogram calculation,
#     # i.e. bins are consecutive integers
#     # Currently, coo_matrix is faster than histogram2d for simple cases
#     # TODO redo it with numpy
#     contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
#                                  (class_idx, cluster_idx)),
#                                 shape=(n_classes, n_clusters),
#                                 dtype=np.int)
#     contingency = contingency.tocsr()
#     contingency.sum_duplicates()
#     return contingency
#
#
# def __mi(U, V):
#     contingency = __contingency_matrix(U, V)
#     nzx, nzy, nz_val = sp.find(contingency)
#     contingency_sum = contingency.sum()
#     pi = np.ravel(contingency.sum(axis=1))
#     pj = np.ravel(contingency.sum(axis=0))
#     log_contingency_nm = np.log(nz_val)
#     contingency_nm = nz_val / contingency_sum
#     # Don't need to calculate the full outer product, just for non-zeroes
#     outer = (pi.take(nzx).astype(np.int64, copy=False)
#              * pj.take(nzy).astype(np.int64, copy=False))
#     log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
#     mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
#           contingency_nm * log_outer)
#     return mi.sum()
#

def spearman_corr(x, y):
    """Calculate Spearman's correlation for each feature. Bigger absolute
    values mean more important features. This measure works best with discrete
    features due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import spearman_corr
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> spearman_corr(x, y)
    array([-0.186339  ,  0.30429031, -0.52704628, -0.30555556,  0.35355339])
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError("The input should contain more than 1 sample")

    x_ranks = np.apply_along_axis(rankdata, 0, x)
    y_ranks = rankdata(y)

    return pearson_corr(x_ranks, y_ranks)


def pearson_corr(x, y):
    """Calculate Pearson's correlation for each feature. Bigger absolute
    values mean more important features. This measure works best with discrete
    features due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import pearson_corr
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> pearson_corr(x, y)
    array([-0.13363062,  0.32732684, -0.60631301, -0.26244533,  0.53452248])
    """
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    sum_dev = y_dev.T.dot(x_dev).reshape((x.shape[1],))
    denominators = np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i
         in range(len(denominators))])
    return results


# TODO need to implement unsupervised way
def laplacian_score(x, y, k_neighbors=5, t=1, metric='euclidean', **kwargs):
    """Calculate Laplacian Score for each feature. Smaller values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    k_neighbors : int, optional
        The number of neighbors to construct a nearest neighbor graph.
    t : float, optional
        Suitable constant for weight matrix S
        where Sij = exp(-(|xi - xj| ^ 2) / t).
    metric : str or callable, optional
        Norm function to compute distance between two points or one of the
        commonly used strings ('euclidean', 'manhattan' etc.) The default
        metric is euclidean.
    weights : array-like, shape (n_samples, n_samples)
        The weight matrix of the graph that models the local structure of
        the data space. By default it is constructed using KNN algorithm.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import laplacian_score
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> laplacian_score(x, y)
    array([1.98983619, 1.22248371,        nan, 0.79710221, 1.90648048])
    """
    n, m = x.shape
    k_neighbors = min(k_neighbors, n - 1)
    if 'weights' in kwargs.keys():
        S = kwargs['weights']
    else:
        if n > 100000:
            S = lil_matrix((n, n))
        else:
            S = np.zeros((n, n))
        graph = NearestNeighbors(n_neighbors=k_neighbors, metric=metric)
        graph.fit(x)
        distances, neighbors = graph.kneighbors()
        for i in range(n):
            for j in range(k_neighbors):
                S[i, neighbors[i][j]] = S[neighbors[i][j], i] = exp(
                    -distances[i][j] * distances[i][j] / t)
    ONE = np.ones((n,))
    D = np.diag(S.dot(ONE))
    L = D - S
    t = D.dot(ONE)
    F = x - x.T.dot(t) / ONE.dot(t)
    F = F.T.dot(L.dot(F)) / F.T.dot(D.dot(F))
    return np.diag(F)


def information_gain(x, y):
    """Calculate mutual information for each feature by formula
    I(X,Y) = H(Y) - H(Y|X). Bigger values mean more important features. This
    measure works best with discrete features due to being based on information
    theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import information_gain
    >>> import numpy as np
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> information_gain(x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    """
    entropy_x = entropy(y)
    cond_entropy = np.apply_along_axis(conditional_entropy, 0, x, y)
    return entropy_x - cond_entropy


def anova(x, y):
    """Calculate anova measure for each feature. Bigger values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    Lowry, Richard.  "Concepts and Applications of Inferential Statistics".
    Chapter 14. http://vassarstats.net/textbook/

    Note:
    The Anova score is counted for checking hypothesis if variances of two
    samples are similar, this measure only returns you counted F-score.
    For understanding whether samples' variances are similar you should
    compare recieved result with value of F-distribution function, for
    example use:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html#scipy.special.fdtrc

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import anova
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 3, 3])
    >>> anova(x, y)
    array([12.6 ,  0.04,   nan,  1.4 ,  3.  ])
    """
    split_by_class = [x[y == k] for k in np.unique(y)]
    num_classes = len(np.unique(y))
    num_samples = x.shape[0]
    num_samples_by_class = [s.shape[0] for s in split_by_class]
    sq_sum_all = sum((s ** 2).sum(axis=0) for s in split_by_class)
    sum_group = [np.asarray(s.sum(axis=0)) for s in split_by_class]
    sq_sum_combined = sum(sum_group) ** 2
    sum_sq_group = [np.asarray((s ** 2).sum(axis=0)) for s in split_by_class]
    sq_sum_group = [s ** 2 for s in sum_group]
    sq_sum_total = sq_sum_all - sq_sum_combined / float(num_samples)
    sq_sum_within = sum(
        [sum_sq_group[i] - sq_sum_group[i] / num_samples_by_class[i] for i in
         range(num_classes)])
    sq_sum_between = sq_sum_total - sq_sum_within
    deg_free_between = num_classes - 1
    deg_free_within = num_samples - num_classes
    ms_between = sq_sum_between / float(deg_free_between)
    ms_within = sq_sum_within / float(deg_free_within)
    f = ms_between / ms_within
    return np.array(f)


def modified_t_score(x, y):
    """Calculate the Modified T-score for each feature. Bigger values mean
    more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples. There can be only 2 classes.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import modified_t_score
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 1, 2, 1, 2])
    >>> modified_t_score(x, y)
    array([1.68968099, 0.12148022, 0.39653932, 0.17682997, 2.04387142])
    """
    classes = np.unique(y)

    size_class0 = y[y == classes[0]].size
    size_class1 = y[y == classes[1]].size

    mean_class0 = np.mean(x[y == classes[0]], axis=0)
    mean_class0 = np.nan_to_num(mean_class0)
    mean_class1 = np.mean(x[y == classes[1]], axis=0)
    mean_class1 = np.nan_to_num(mean_class1)

    std_class0 = np.std(x[y == classes[0]], axis=0)
    std_class0 = np.nan_to_num(std_class0)
    std_class1 = np.std(x[y == classes[1]], axis=0)
    std_class1 = np.nan_to_num(std_class1)

    corr_with_y = np.apply_along_axis(
        lambda feature: abs(np.corrcoef(feature, y)[0][1]), 0, x)
    corr_with_y = np.nan_to_num(corr_with_y)

    corr_with_others = abs(np.corrcoef(x, rowvar=False))
    corr_with_others = np.nan_to_num(corr_with_others)

    mean_of_corr_with_others = (
        corr_with_others.sum(axis=1)
        - corr_with_others.diagonal()) / (len(corr_with_others) - 1)

    t_score_numerator = abs(mean_class0 - mean_class1)
    t_score_denominator = np.sqrt(
        (size_class0 * np.square(std_class0) + size_class1 * np.square(
            std_class1)) / (size_class0 + size_class1))
    modificator = corr_with_y / mean_of_corr_with_others

    modified_t_score = t_score_numerator / t_score_denominator * modificator
    modified_t_score = np.nan_to_num(modified_t_score)

    return modified_t_score


MEASURE_NAMES = {"FitCriterion": fit_criterion_measure,
                 "FRatio": f_ratio_measure,
                 "GiniIndex": gini_index,
                 "SymmetricUncertainty": su_measure,
                 "SpearmanCorr": spearman_corr,
                 "PearsonCorr": pearson_corr,
                 "FechnerCorr": fechner_corr,
                 "KendallCorr": kendall_corr,
                 "ReliefF": reliefF_measure,
                 "Chi2": chi2_measure,
                 "Anova": anova,
                 "LaplacianScore": laplacian_score,
                 "InformationGain": information_gain,
                 "ModifiedTScore": modified_t_score,
                 "Relief": relief_measure}


def select_best_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=True)


def select_worst_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=False)


def __select_by_value(scores, value, more=True):
    if more:
        return np.flatnonzero(scores >= value)
    else:
        return np.flatnonzero(scores <= value)


def select_k_best(k):
    return _wrapped_partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return _wrapped_partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if not isinstance(k, int):
        raise TypeError("Number of features should be integer")
    if k > scores.shape[0]:
        raise ValueError(
            "Cannot select %d features with n_features = %d" % (k, len(scores)))
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]
    return order[:k]


def __select_percentage_best(scores, percent):
    return __select_k(
        scores, k=(int)(scores.shape[0] * percent), reverse=True)


def select_best_percentage(percent):
    return _wrapped_partial(__select_percentage_best, percent=percent)


def __select_percentage_worst(scores, percent):
    return __select_k(
        scores, k=(int)(scores.shape[0] * percent), reverse=False)


def select_worst_percentage(percent):
    return _wrapped_partial(__select_percentage_worst, percent=percent)


CR_NAMES = {"Best by value": select_best_by_value,
            "Worst by value": select_worst_by_value,
            "K best": select_k_best,
            "K worst": select_k_worst,
            "Worst by percentage": select_worst_percentage,
            "Best by percentage": select_best_percentage}


def qpfs_filter(X, y, r=None, sigma=None, solv='quadprog', fn=pearson_corr):
    """Performs Quadratic Programming Feature Selection algorithm.
    Note: this realization requires labels to start from 1 and be numerical.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    r : int
        The number of samples to be used in Nystrom optimization.
    sigma : double
        The threshold for eigenvalues to be used in solving QP optimization.
    solv : string, default
        The name of qp solver according to
        qpsolvers(https://pypi.org/project/qpsolvers/) naming. Note quadprog
        is used by default.
    fn : function(array, array), default
        The function to count correlation, for example pierson correlation or
        mutual information. Note mutual information is used by default.

    Returns
    -------
    array-like, shape (n_features,) : the ranks of features in dataset, with
    rank increase, feature relevance increases and redundancy decreases.

    See Also
    --------
    http://www.jmlr.org/papers/volume11/rodriguez-lujan10a/rodriguez-lujan10a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import qpfs_filter
    >>> from sklearn.datasets import make_classification
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> ranks = qpfs_filter(x, y)
    >>> print(ranks)
    """
    return qpfs_body(X, y, fn, r=r, sigma=sigma, solv=solv)
