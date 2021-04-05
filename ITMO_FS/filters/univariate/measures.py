from functools import partial, update_wrapper
from math import exp
from math import log

import numpy as np
from scipy import sparse as sp
from scipy.sparse import lil_matrix
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from ITMO_FS.utils.data_check import generate_features
from ITMO_FS.utils.information_theory import conditional_entropy
from ITMO_FS.utils.information_theory import entropy
from ITMO_FS.utils.qpfs_body import qpfs_body


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def fit_criterion_measure(x, y):
    """
    Calculates the FitCriterion score for features. Bigger values mean more 
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples, )
        The target values.

    Returns
    -------
    array-like, shape (n_features) : feature scores
        
    Notes ----- For more details see `this paper
    <https://core.ac.uk/download/pdf/191234514.pdf/>`_.


    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fit_criterion_measure
    >>> import numpy as np
    >>> x = np.array([[1, 2, 4, 1, 1],[2, 2, 2, 1, 2], [3, 5, 1, 1, 4], \
[1, 1, 1, 1, 4],[2, 2, 2, 1, 5]], dtype = np.integer)
    >>> y = np.array([1, 2, 3, 1, 2], dtype=np.integer)
    >>> fit_criterion_measure(x, y)
    array([1. , 0.8, 0.8, 0.4, 0.6])
    """
    def count_hits(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        means = {cl: np.mean(splits[cl]) for cl in classes}
        devs = {cl: np.var(splits[cl]) for cl in classes}
        distances = np.vectorize(lambda x_val: {cl: abs(x_val - means[cl]) / 
            (devs[cl] + 1e-10) for cl in classes})(feature)
        return np.sum(np.vectorize(lambda d: min(d, key=d.get))(distances) == y)

    classes = np.unique(y)
    return np.apply_along_axis(count_hits, 0, x) / x.shape[0]


def f_ratio_measure(x, y):
    """
    Calculates Fisher score for features. Bigger values mean more 
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import f_ratio_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> f_ratio_measure(x, y)
    array([0.6 , 0.2 , 1.  , 0.12, 5.4 ])
    """
    def __F_ratio(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        mean_feature = np.mean(feature)
        inter_class = np.sum(np.vectorize(lambda cl: counts_d[cl] * 
            np.power(mean_feature - np.mean(splits[cl]), 2))(classes))
        intra_class = np.sum(np.vectorize(lambda cl: counts_d[cl] * 
            np.var(splits[cl]))(classes))
        return inter_class / (intra_class + 1e-10)

    classes, counts = np.unique(y, return_counts=True)
    counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}
    return np.apply_along_axis(__F_ratio, 0, x)


def gini_index(x, y):
    """
    Calculates Gini index for features. Bigger values mean more important 
    features. This measure works best with discrete features due to being based 
    on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import gini_index
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> gini_index(x, y)
    array([0.14      , 0.04      , 0.64      , 0.24      , 0.37333333])
    """

    def __gini(feature):
        values, counts = np.unique(feature, return_counts=True)
        counts_d = {val: counts[idx] for idx, val in enumerate(values)}
        return np.sum(np.vectorize(lambda val: np.sum(
            np.square(np.unique(y[feature == val], return_counts=True)[1])) /
            counts_d[val])(values)) / x.shape[0] - prior_prob_squared_sum

    classes, counts = np.unique(y, return_counts=True)
    prior_prob_squared_sum = np.sum(np.square(counts / x.shape[0]))
    
    return np.apply_along_axis(__gini, 0, x)
    


def su_measure(x, y):
    """
    SU is a correlation measure between the features and the class
    calculated, via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y)).
    Bigger values mean more important features. This measure works
    best with discrete features due to being based on information
    theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import su_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> su_measure(x, y)
    array([0.28694182, 0.13715115, 0.79187567, 0.47435099, 0.67126949])
    """
    def __SU(feature):
        entropy_x = entropy(feature)
        return 2 * (entropy_x - conditional_entropy(y, feature)) / (
            entropy_x + entropy_y)

    entropy_y = entropy(y)
    return np.apply_along_axis(__SU, 0, x)


# TODO CONCORDATION COEF

def kendall_corr(x, y):
    """
    Calculates Sample sign correlation (Kendall correlation) for each feature.
    Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features) or (n_samples, )
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import kendall_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> kendall_corr(x, y)
    array([-0.1,  0.2, -0.4, -0.2,  0.2])
    """

    def __kendall_corr(feature):
        k_corr = 0.0
        for i in range(len(feature)):
            for j in range(i + 1, len(feature)):
                k_corr += np.sign(feature[i] - feature[j]) * np.sign(y[i] - y[j])
        return 2 * k_corr / (feature.shape[0] * (feature.shape[0] - 1))

    return np.apply_along_axis(__kendall_corr, 0, x)


def fechner_corr(x, y):
    """
    Calculates Sample sign correlation (Fechner correlation) for each feature.
    Bigger absolute values mean more important features.    

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fechner_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> fechner_corr(x, y)
    array([-0.2,  0.2, -0.4, -0.2, -0.2])
    """

    y_dev = y - np.mean(y)
    x_dev = x - np.mean(x, axis=0)
    return np.sum(np.sign(x_dev.T * y_dev), axis=1) / x.shape[0]


def __distance_matrix(X, y, n_samples):
    dm = np.zeros((n_samples, n_samples), dtype=tuple)
    for i in range(n_samples):
        for j in range(i, n_samples):
            # using the Manhattan (L1) norm rather than
            # the Euclidean (L2) norm,
            # although the rationale is not specified
            value = np.linalg.norm(X[i, :] - X[j, :], 1)
            dm[i, j] = (value, j, y[j])
            dm[j, i] = (value, i, y[i])
    # sort_indices = dm.argsort(1)
    # dm.sort(1)
    # indices = np.arange(n_samples) #[sort_indices]
    # dm = np.dstack((dm, indices))
    return dm

    # TODO redo with np.where


def __take_k(dm_i, k, r_index, choice_func):
    hits = []
    dm_i = sorted(dm_i, key=lambda x: x[0])
    for samp in dm_i:
        if (samp[1] != r_index) & (k > 0) & (choice_func(samp[2])):
            hits.append(samp)
            k -= 1
    return np.array(hits, int)


def reliefF_measure(x, y, k_neighbors=1):
    """
    Counts ReliefF measure for each feature. Bigger values mean more important 
    features. 

    Note:
    Only for complete x
    Rather than repeating the algorithm m(TODO Ask Nikita about user defined) 
    times, implement it exhaustively (i.e. n times, once for each instance)
    for relatively small n (up to one thousand).

    Calculates spearman correlation for each feature.
    Spearman's correlation assesses monotonic relationships (whether linear or 
    not). If there are no repeated data values, a perfect Spearman correlation 
    of +1 or −1 occurs when each of the variables is a perfect monotone function 
    of the other.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.
    k_neighbors : int, optional
        The number of neighbors to consider when assigning feature importance 
        scores. More neighbors results in more accurate scores, but takes longer.
        Selection of k hits and misses is the basic difference to Relief
        and ensures greater robustness of the algorithm concerning noise.


    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and review
    Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import reliefF_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> reliefF_measure(x, y)
    array([-0.2       , -0.43333333,  0.075     , -0.31666667,  0.31666667])
    """
    f_ratios = np.zeros(x.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    prior_prob = dict(zip(classes, np.array(counts) / len(y)))
    n_samples = x.shape[0]
    n_features = x.shape[1]
    dm = __distance_matrix(x, y, n_samples)
    for i in range(n_samples):
        r = x[i]
        dm_i = dm[i]
        hits = __take_k(dm_i, k_neighbors, i, lambda x: x == y[i])
        if len(hits) != 0:
            ind_hits = hits[:, 1]
        else:
            ind_hits = []
        value_hits = x.take(ind_hits, axis=0)
        m_c = np.empty(len(classes), np.ndarray)
        for j in range(len(classes)):
            if classes[j] != y[i]:
                misses = __take_k(dm_i, k_neighbors, i,
                                  lambda x: x == classes[j])
                ind_misses = misses[:, 1]
                m_c[j] = x.take(ind_misses, axis=0)
        for A in range(n_features):
            weight_hit = np.sum(np.abs(r[A] - value_hits[:, A]))
            weight_miss = 0
            for j in range(len(classes)):
                if classes[j] != y[i]:
                    weight_miss += prior_prob[classes[j]] * np.sum(
                        np.abs(r[A] - m_c[j][:, A]))
            f_ratios[A] += weight_miss / (1 - prior_prob[y[i]]) - weight_hit
    # dividing by m * k guarantees that all final weights
    # will be normalized within the interval [ − 1, 1].
    f_ratios /= n_samples * k_neighbors
    # The maximum and minimum values of A are determined over the entire
    # set of instances.
    # This normalization ensures that weight updates fall
    # between 0 and 1 for both discrete and continuous features.
    with np.errstate(divide='ignore', invalid="ignore"):  # todo
        return f_ratios / (np.amax(x, axis=0) - np.amin(x, axis=0))


def relief_measure(x, y, m=None, random_state=42):
    """
    Computes Relief measure for each feature. This measure is supposed to work 
    only with binary classification datasets; for multi-class problems use the 
    ReliefF measure. Bigger values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.
    m : int, optional
        Amount of iterations to do. If not specified, n_samples iterations would 
        be performed.
    random_state : int, optional
        Random state for numpy random.
    
    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and review
    Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import relief_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> relief_measure(x, y)
    array([ 0.    , -0.6   , -0.1875, -0.15  , -0.4   ])
    """
    weights = np.zeros(x.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 1:
        raise ValueError('Cannot calculate relief measure with 1 class')

    n_samples = x.shape[0]
    n_features = x.shape[1]
    if m is None:
        m = n_samples

    x_normalized = MinMaxScaler().fit_transform(x)
    dm = euclidean_distances(x_normalized, x_normalized)
    indices = np.random.default_rng(random_state).integers(low=0, 
        high=n_samples, size=m) 
    for i in indices:
        distances = dm[i]
        order = np.argsort(distances)[1:]

        for index in order:
            if y[index] == y[i]:
                weights -= np.square(x_normalized[i] - x_normalized[index])
                break

        for index in order:
            if y[index] != y[i]:
                weights += np.square(x_normalized[i] - x_normalized[index])
                break

    return weights / m


def chi2_measure(x, y):
    """
    Calculates the Chi-squared measure for each feature. Bigger values mean more 
    important features. This measure works best with discrete features due to 
    being based on statistics. 
    
    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Example
    -------
    >>> from ITMO_FS.filters.univariate import chi2_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
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
        e = np.vectorize(lambda cl: prior_probs[cl] * counts, 
            signature='()->(1)')(classes)
        n = np.vectorize(lambda cl: np.bincount(splits[cl], 
            minlength=values.shape[0]), signature='()->(1)')(classes)
        return np.sum(np.square(e - n) / e)

    classes, counts = np.unique(y, return_counts=True)
    prior_probs = {cl: counts[idx] / x.shape[0] for idx, cl in enumerate(classes)}
    
    return np.apply_along_axis(__chi2, 0, x)


def __contingency_matrix(labels_true, labels_pred):
    """Build a contingency matrix describing the relationship between labels.
        Parameters
        ----------
        labels_true : int array, shape = [n_samples]
            Ground truth class labels to be used as a reference
        labels_pred : array, shape = [n_samples]
            Cluster labels to evaluate
        Returns
        -------
        contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
            Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
            true class :math:`i` and in predicted class :math:`j`. If
            ``eps is None``, the dtype of this array will be integer. If ``eps`` is
            given, the dtype will be float.
        """
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    # TODO redo it with numpy
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    contingency = contingency.tocsr()
    contingency.sum_duplicates()
    return contingency


def __mi(U, V):
    contingency = __contingency_matrix(U, V)
    nzx, nzy, nz_val = sp.find(contingency)
    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = (pi.take(nzx).astype(np.int64, copy=False)
             * pj.take(nzy).astype(np.int64, copy=False))
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
          contingency_nm * log_outer)
    return mi.sum()


def spearman_corr(x, y):
    """
    Calculates Spearman's correlation for each feature. Bigger absolute values 
    mean more important features. This measure works best with discrete features 
    due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import spearman_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
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
    """
    Calculates Pearson's correlation for each feature. Bigger absolute values 
    mean more important features. This measure works best with discrete features 
    due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import pearson_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> pearson_corr(x, y)
    array([-0.13363062,  0.32732684, -0.56694671, -0.28571429,  0.53452248])
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
    """
    Calculates Laplacian Score for each feature. Smaller values mean more 
    important features.

    Parameters
    ----------
    x : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.
    k_neighbors : int, optional (by default k_neighbors=5)
        The number of neighbors to construct a nearest neighbor graph.
    t : float, optional
        Suitable constant for weight matrix S, 
        where Sij = exp(-(|xi - xj| ^ 2) / t).
    metric : str or callable, optional
        Norm function to compute distance between two points or one of the 
        commonly used strings ('euclidean', 'manhattan' etc.) The default 
        metric is euclidean.
    weights : array-like, shape (n_samples, n_samples)
        The weight matrix of the graph that models the local structure of the 
        data space. By default it is constructed using KNN algorithm.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import laplacian_score
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3], \
        [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]], dtype=np.integer)
    >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
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
    """
    Calculates mutual information for each feature by formula,
    I(X,Y) = H(Y) - H(Y|X). Bigger values mean more important features. This 
    measure works best with discrete features due to being based on information 
    theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import information_gain
    >>> import numpy as np
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3], \
        [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]], dtype=np.integer)
    >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> information_gain(x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    """
    entropy_x = entropy(y)
    cond_entropy = np.apply_along_axis(conditional_entropy, 0, x, y)
    return entropy_x - cond_entropy


def anova(x, y):
    """
    Calculates anova measure for each feature. Bigger values mean more important
    features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    Lowry, Richard.  "Concepts and Applications of Inferential
    Statistics". Chapter 14.
    http://vassarstats.net/textbook/

    Note:
    The Anova score is counted for checking hypothesis if variances of two 
    samples are similar, this measure only returns you counted F-score.
    For understanding whether samples' variances are similar you should compare 
    recieved result with value of F-distribution function, for example use:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html#scipy.special.fdtrc

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import anova
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3], \
        [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]], dtype=np.integer)
    >>> y = np.array([1, 2, 1, 3, 3], dtype=np.integer)
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
    """
    Calculate the Modified T-score for each feature. Bigger values mean more 
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples, )
        The classes for the samples. There can be only 2 classes.

    Returns
    -------
    array-like, shape (n_features) : feature scores

    See Also
    --------
    For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import modified_t_score
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
        [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
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

    mean_of_corr_with_others = (corr_with_others.sum(
        axis=1) - corr_with_others.diagonal()) / (
        len(corr_with_others) - 1)

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
    features = []
    for key, sc_value in scores.items():
        if more:
            if sc_value >= value:
                features.append(key)
        else:
            if sc_value <= value:
                features.append(key)
    return features


def select_k_best(k):
    return _wrapped_partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return _wrapped_partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if not isinstance(k, int):
        raise TypeError("Number of features should be integer")
    if k > len(scores):
        raise ValueError("Cannot select %d features with n_features = %d" % (
            k, len(scores)))
    return [keys[0] for keys in
            sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


def __select_percentage_best(scores, percent):
    features = []
    max_val = max(scores.values())
    threshold = max_val * percent
    for key, sc_value in scores.items():
        if sc_value >= threshold:
            features.append(key)
    return features


def select_best_percentage(percent):
    return _wrapped_partial(__select_percentage_best, percent=percent)


def __select_percentage_worst(scores, percent):
    features = []
    max_val = min(scores.values())
    threshold = max_val * percent
    for key, sc_value in scores.items():
        if sc_value >= threshold:
            features.append(key)
    return features


def select_worst_percentage(percent):
    return _wrapped_partial(__select_percentage_worst, percent=percent)


CR_NAMES = {"Best by value": select_best_by_value,
           "Worst by value": select_worst_by_value,
           "K best": select_k_best,
           "K worst": select_k_worst,
           "Worst by percentage": select_worst_percentage,
           "Best by percentage": select_best_percentage}


def qpfs_filter(X, y, r=None, sigma=None, solv='quadprog', fn=pearson_corr):
    """
    Performs Quadratic Programming Feature Selection algorithm.
    Note: this realization requires labels to start from 1 and be numerical.

    Parameters
    ----------
    X : array-like, shape (n_samples,n_features)
        The input samples.
    y : array-like, shape (n_samples)
        The classes for the samples.
    r : int
        The number of samples to be used in Nystrom optimization.
    sigma : double
        The threshold for eigenvalues to be used in solving QP optimization.
    solv : string, default
        The name of qp solver according to qpsolvers(https://pypi.org/project/qpsolvers/) naming.
        Note quadprog is used by default.
    fn : function(array, array), default
        The function to count correlation, for example pierson correlation or  mutual information.
        Note mutual information is used by default.
    Returns
    ------
    array-like, shape (n_features) : the ranks of features in dataset, with rank increase, feature relevance increases and redundancy decreases.

    See Also
    --------
    http://www.jmlr.org/papers/volume11/rodriguez-lujan10a/rodriguez-lujan10a.pdf

    Examples
    --------

    >>> from ITMO_FS.filters.univariate import qpfs_filter
    >>> from sklearn.datasets import make_classification
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> ranks = qpfs_filter(x, y)
    >>> print(ranks)

    """

    return qpfs_body(X, y, fn, r=r, sigma=sigma, solv=solv)
