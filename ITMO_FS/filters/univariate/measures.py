from functools import partial, update_wrapper
from math import exp
from math import log

import numpy as np
from scipy import sparse as sp
from scipy.sparse import lil_matrix
from sklearn.preprocessing import MinMaxScaler

from ITMO_FS.utils.data_check import generate_features
from ITMO_FS.utils.information_theory import conditional_entropy
from ITMO_FS.utils.information_theory import entropy
from ITMO_FS.utils.qpfs_body import qpfs_body


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def fit_criterion_measure(X, y):
    ##todo comments
    x = np.asarray(X)  # Converting input data to numpy array
    y = np.asarray(y.reshape((-1,)))
    if len(x.shape) == 2:
        fc = np.zeros(x.shape[1])  # Array with amounts of correct predictions for each feature
    else:
        fc = np.zeros(len(x))
    tokens_n = np.unique(y)  # Number of different class tokens
    centers = np.empty(tokens_n)  # Array with centers of sets of feature values for each class token
    variances = np.empty(tokens_n)  # Array with variances of sets of feature values for each class token
    # Each of arrays above will be separately calculated for each feature
    distances = np.empty(tokens_n)  # Array with distances between sample's value and each class's center
    # This array will be separately calculated for each feature and each sample
    for feature_index, feature in enumerate(x.T):  # For each feature
        # Initializing utility structures
        class_values = [[] for _ in range(tokens_n)]  # Array with lists of feature values for each class token
        for index, value in enumerate(y):  # Filling array
            class_values[value].append(feature[index])
        for token, values in enumerate(class_values):  # For each class token's list of feature values
            tmp_arr = np.array(values)
            centers[token] = np.mean(tmp_arr)
            variances[token] = np.var(tmp_arr)
        # Main calculations
        for sample_index, value in enumerate(feature):  # For each sample value
            for i in range(tokens_n):  # For each class token
                # Here can be raise warnings by 0/0 division. In this case, default results
                # are interpreted correctly
                distances[i] = np.abs(value - centers[i]) / variances[i]
            fc[feature_index] += np.argmin(distances) == y[sample_index]
    fc /= y.shape[0]
    return dict(zip(generate_features(x), fc))


def __calculate_F_ratio(row, y_data):
    inter_class = 0.0
    intra_class = 0.0
    mean_feature = np.mean(row)
    for value in np.unique(y_data):
        index_for_this_value = np.where(y_data == value)[0]
        n = np.sum(row[index_for_this_value])
        mu = np.mean(row[index_for_this_value])
        var = np.var(row[index_for_this_value])
        inter_class += n * np.power((mu - mean_feature), 2)
        intra_class += (n - 1) * var
    if inter_class == 0. and intra_class == 0.:
        return 0.
    elif intra_class == 0.:
        return float('inf')
    else:
        return inter_class / intra_class


def f_ratio_measure(X, y):
    """
    Calculates Fisher score for features.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import f_ratio_measure
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = f_ratio_measure(X, y)
    >>> print(scores)
    [ 0.96        0.37333333  0.5         0.1725     13.26      ]
    """
    return np.apply_along_axis(__calculate_F_ratio, 0, X, y)


def gini_index(X, y):
    """
    Gini index is a measure of statistical dispersion.
    Note: before counting gini index data is normalized with MinMaxScaler

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://en.wikipedia.org/wiki/Gini_coefficient
    
    Examples
    --------
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import gini_index

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = gini_index(X, y)
    print(scores)

    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import gini_index
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = gini_index(X, y)
    >>> print(scores)
    [0.05555556 0.44444444 0.05555556 0.16666667 0.62962963]
    """

    if X.shape[0] < 2:
        raise ValueError("The input should contain more than 1 sample")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    cum_x = np.cumsum(X / np.linalg.norm(X, 1, axis=0), axis=0)
    cum_y = np.cumsum(y / np.linalg.norm(y, 1))
    diff_x = (cum_x[1:] - cum_x[:-1])
    diff_y = (cum_y[1:] + cum_y[:-1])
    return np.abs(1 - np.sum(np.multiply(diff_x.T, diff_y).T, axis=0))


def su_measure(X, y):
    """
    SU is a correlation measure between the features and the class
    calculated, via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y))

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import su_measure
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = su_measure(X, y)
    >>> print(scores)
    [0.82173546 0.67908587 0.79187567 0.73717549 0.86172942]
    """
    entropy_y = entropy(y)
    f_ratios = np.empty(X.shape[1])
    for index in range(X.shape[1]):
        entropy_x = entropy(X[:, index])
        cond_entropy = conditional_entropy(y, X[:, index])
        f_ratios[index] = 2 * (entropy_x - cond_entropy) / (entropy_x + entropy_y)
    return f_ratios


# TODO CONCORDATION COEF

def _kendall_corr(X, y):
    k_corr = 0.0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            k_corr += np.sign(X[i] - X[j]) * np.sign(y[i] - y[j])
    return k_corr


def kendall_corr(X, y):
    """
    Calculates Sample sign correlation (Kendall correlation) for each feature.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features) or (n_samples, )
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import kendall_corr
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = kendall_corr(X, y)
    >>> print(scores)
    [-0.1  0.2 -0.4 -0.2  0.2]
    """

    if len(X.shape) == 1:
        k_corr = _kendall_corr(X, y)
        return float(2 * k_corr) / (len(X) * (len(X) - 1))
    else:
        res = []
        for var in range(X.shape[1]):
            x = X[:, var]
            k_corr = _kendall_corr(x, y)
            k_corr = float(2 * k_corr) / (len(x) * (len(x) - 1))
            res.append(k_corr)
        return np.array(res)


def fechner_corr(X, y):
    """
    Calculates Sample sign correlation (Fechner correlation) for each feature.
    

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import fechner_corr
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = fechner_corr(X, y)
    >>> print(scores)
    [-0.2  0.2 -0.5 -0.2 -0.2]
    """

    if len(X.shape) == 1:
        m = 1
    else:
        m = X.shape[1]
    y_dev = y - np.mean(y)
    x_dev = X - np.mean(X, axis=0)
    if m == 1:
        N_plus = np.array([np.sum((x_dev >= 0) & (y_dev >= 0)) + np.sum((x_dev < 0) & (y_dev < 0))])
        N_minus = np.array([np.sum((x_dev > 0) & (y_dev < 0)) + np.sum((x_dev < 0) & (y_dev > 0))])
    else:
        N_plus = np.sum((x_dev >= 0).T & (y_dev >= 0), axis=1) + np.sum((x_dev < 0).T & (y_dev < 0), axis=1).astype(
            float)
        N_minus = np.sum((x_dev > 0).T & (y_dev < 0), axis=1) + np.sum((x_dev < 0).T & (y_dev > 0), axis=1).astype(
            float)
    return (N_plus - N_minus) / (N_plus + N_minus)


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


def reliefF_measure(X, y, k_neighbors=1):
    """
    Counts ReliefF measure for each feature

    Note:
    Only for complete X
    Rather than repeating the algorithm m(TODO Ask Nikita about user defined) times,
    implement it exhaustively (i.e. n times, once for each instance)
    for relatively small n (up to one thousand).
    
    Calculates spearman correlation for each feature.
    Spearman's correlation assesses monotonic relationships (whether linear or not).
    If there are no repeated data values, a perfect Spearman correlation of +1 or −1
    occurs when each of the variables is a perfect monotone function of the other.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.
    k_neighbors : int, optional = 1,
        The number of neighbors to consider when assigning feature importance scores.
        More neighbors results in more accurate scores, but takes longer.
        Selection of k hits and misses is the basic difference to Relief
        and ensures greater robustness of the algorithm concerning noise.
    

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and review
    Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import reliefF_measure
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = reliefF_measure(X, y)
    >>> print(scores)
    [-0.13333333 -0.41666667  0.17916667 -0.275       0.46666667]
    """
    f_ratios = np.zeros(X.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    prior_prob = dict(zip(classes, np.array(counts) / len(y)))
    n_samples = X.shape[0]
    n_features = X.shape[1]
    dm = __distance_matrix(X, y, n_samples)
    for i in range(n_samples):
        r = X[i]
        dm_i = dm[i]
        hits = __take_k(dm_i, k_neighbors, i, lambda x: x == y[i])
        if len(hits) != 0:
            ind_hits = hits[:, 1]
        else:
            ind_hits = []
        value_hits = X.take(ind_hits, axis=0)
        m_c = np.empty(len(classes), np.ndarray)
        for j in range(len(classes)):
            if classes[j] != y[i]:
                misses = __take_k(dm_i, k_neighbors, i, lambda x: x == classes[j])
                ind_misses = misses[:, 1]
                m_c[j] = X.take(ind_misses, axis=0)
        for A in range(n_features):
            weight_hit = np.sum(np.abs(r[A] - value_hits[:, A]))
            weight_miss = 0
            for j in range(len(classes)):
                if classes[j] != y[i]:
                    weight_miss += prior_prob[y[j]] * np.sum(np.abs(r[A] - m_c[j][:, A]))
            f_ratios[A] += weight_miss / (1 - prior_prob[y[i]]) - weight_hit
    # dividing by m * k guarantees that all final weights
    # will be normalized within the interval [ − 1, 1].
    f_ratios /= n_samples * k_neighbors
    # The maximum and minimum values of A are determined over the entire
    # set of instances.
    # This normalization ensures that weight updates fall
    # between 0 and 1 for both discrete and continuous features.
    with np.errstate(divide='ignore', invalid="ignore"):  # todo
        return f_ratios / (np.amax(X, axis=0) - np.amin(X, axis=0))


def __label_binarize(y):
    """
    Binarize labels in a one-vs-all fashion
    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.
    """
    classes = np.unique(y)
    n_samples = len(y)
    n_classes = len(classes)
    row = np.arange(n_samples)
    col = [np.where(classes == el)[0][0] for el in y]
    data = np.repeat(1, n_samples)
    # TODO redo it with numpy
    return sp.csr_matrix((data, (row, col)), shape=(n_samples, n_classes)).toarray()


def __chisquare(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.
    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return chisq


def chi2_measure(X, y):
    """
    Calculates score for the test chi-squared statistic from X.
    Chi-squared test is a statistical hypothesis test that
    is valid to perform when the test statistic is chi-squared 
    distributed under the null hypothesis

    Note: Input data must contain only non-negative features such
    as booleans or frequencies (e.g., term counts in document classification),
    relative to the classes.
    
    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    ##TODO examples
    --------
    """

    if np.any(X < 0):
        raise ValueError("Input X must be non-negative.")
    y = __label_binarize(y)
    # If you use sparse input
    # you can use sklearn.utils.extmath.safe_sparse_dot instead
    observed = np.dot(y.T, X)  # n_classes * n_features
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)
    return __chisquare(observed, expected)


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


def spearman_corr(X, y):
    """
    Calculates spearman correlation for each feature.
    Spearman's correlation assesses monotonic relationships (whether linear or not).
    If there are no repeated data values, a perfect Spearman correlation of +1 or −1
    occurs when each of the variables is a perfect monotone function of the other.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import spearman_corr
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> scores = spearman_corr(X, y)
    >>> scores
    array([-0.4 ,  0.3 , -0.3 , -0.2 ,  0.65])
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("The input should contain more than 1 sample")
    c = 6 / (n * (n - 1) * (n + 1))

    d = dict(zip(sorted(y), range(y.shape[0])))
    ranks_y = np.vectorize(d.get)(y)
    if y.shape == X.shape:
        d = dict(zip(sorted(X), range(X.shape[0])))
        ranks_X = np.vectorize(d.get)(X)

        dif = ranks_X - ranks_y
    else:
        ranks_X = X.copy()
        for i in range(X.shape[1]):
            d = dict(zip(sorted(X[:, i]), range(X.shape[0])))
            ranks_X[:, i] = np.vectorize(d.get)(X[:, i])

        dif = ranks_X - np.repeat(ranks_y, X.shape[1]).reshape(y.shape[0], X.shape[1])

    return 1 - c * np.sum(dif * dif, axis=0)


def pearson_corr(X, y):
    """
    Calculates pearson correlation for each feature.
    Pearson correleation coeficient is a statistic
    that measures linear correlation between two variables X and Y.
    It has a value in interval [-1, +1], where 1 is total positive linear correlation,
    0 is no linear correlation, and −1 is total negative linear correlation

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import pearson_corr
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> pearson_corr(X, y)
    array([-0.13363062,  0.32732684, -0.56694671, -0.28571429,  0.53452248])
    """
    x_dev = X - np.mean(X, axis=0)
    y_dev = y - np.mean(y, axis=0)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    if len(X.shape) == 1:
        sum_dev = y_dev.T.dot(x_dev).reshape((1,))
        denominators = np.array([np.sqrt(np.sum(sq_dev_y, axis=0) * np.sum(sq_dev_x, axis=0))])
    else:
        sum_dev = y_dev.T.dot(x_dev).reshape((X.shape[1],))
        denominators = np.sqrt(np.sum(sq_dev_y, axis=0) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i in range(len(denominators))])
    return results


# TODO need to implement unsupervised way
def laplacian_score(X, y, k_neighbors=5, t=1, metric=np.linalg.norm, **kwargs):
    """
    Calculates Laplacian Score for each feature.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.
    k_neighbors : int, optional (by default k_neighbors=5)
        The number of neighbors to construct a nearest neighbor graph.
    t : float, optional (by default t=1)
        Suitable constant for weight matrix S, 
        where Sij = exp(-(|xi - xj| ^ 2) / t).
    metric : callable, optional (by default metric=np.linalg.norm)
        Norm function to compute distance between two points.
        The default distance is euclidean.
    weights : numpy array, shape (n_samples, n_samples)
        The weight matrix of the graph that models the local structure of the data space.
        By default it is constructed using KNN algorithm.

    Returns
    -------
    List of scores of each feature.
    The smaller the laplacian score is, the more important the feature is.

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import laplacian_score
    >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
    >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    >>> scores = laplacian_score(X, y)
    >>> scores
    array([1.98983619, 1.22248371,        nan, 0.79710221, 1.90648048])
    """
    n, m = X.shape
    k_neighbors = min(k_neighbors, n - 1)
    if 'weights' in kwargs.keys():
        S = kwargs['weights']
    else:
        if n > 100000:
            S = lil_matrix((n, n))
        else:
            S = np.zeros((n, n))
        for i in range(n):
            distances = []
            for j in range(n):
                if i == j:
                    continue
                d = metric(X[i] - X[j])
                distances.append((d, j))
                if y[i] == y[j]:
                    S[i, j] = exp(-d * d / t)
            distances.sort()
            for j in range(k_neighbors):
                S[i, distances[j][1]] = S[distances[j][1], i] = exp(-distances[j][0] * distances[j][0] / t)
    ONE = np.ones((n,))
    D = np.diag(S.dot(ONE))
    L = D - S
    t = D.dot(ONE)
    F = X - X.T.dot(t) / ONE.dot(t)
    F = F.T.dot(L.dot(F)) / F.T.dot(D.dot(F))
    return np.diag(F)


def information_gain(X, y):
    """
    Calculates mutual information for each feature by formula,
    I(X,Y) = H(X) - H(X|Y)

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    
    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import information_gain
    >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
    >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    >>> scores = information_gain(X, y)
    >>> print(scores)
    [1.33217904 1.33217904 0.         0.67301167 1.60943791]
    """
    entropy_x = entropy(y)
    cond_entropy = np.apply_along_axis(conditional_entropy, 0, X, y)
    return entropy_x - cond_entropy


def anova(X, y):
    """
    Calculates anova measure for each feature.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, )

    See Also
    --------
    Lowry, Richard.  "Concepts and Applications of Inferential
    Statistics". Chapter 14.
    http://vassarstats.net/textbook/

    Note:
    The Anova score is counted for checking hypothesis if variances of two samples are similar,
    this measure only returns you counted F-score.
    For understanding whether samples' variances are similar you should compare recieved result with 
    value of F-distribution function, for example use: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html#scipy.special.fdtrc

    Examples
    --------
    >>> import sklearn.datasets as datasets
    >>> from ITMO_FS.filters.univariate import anova
    >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
    >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    >>> scores = anova(X, y)
    >>> print(scores)
    """
    split_by_class = [X[y == k] for k in np.unique(y)]
    num_classes = len(np.unique(y))
    num_samples = X.shape[0]
    num_samples_by_class = [s.shape[0] for s in split_by_class]
    sq_sum_all = sum((s ** 2).sum(axis=0) for s in split_by_class)
    sum_group = [np.asarray(s.sum(axis=0)) for s in split_by_class]
    sq_sum_combined = sum(sum_group) ** 2
    sum_sq_group = [np.asarray((s ** 2).sum(axis=0)) for s in split_by_class]
    sq_sum_group = [s ** 2 for s in sum_group]
    sq_sum_total = sq_sum_all - sq_sum_combined / float(num_samples)
    sq_sum_within = sum([sum_sq_group[i] - sq_sum_group[i] / num_samples_by_class[i] for i in range(num_classes)])
    sq_sum_between = sq_sum_total - sq_sum_within
    deg_free_between = num_classes - 1
    deg_free_within = num_samples - num_classes
    ms_between = sq_sum_between / float(deg_free_between)
    ms_within = sq_sum_within / float(deg_free_within)
    f = ms_between / ms_within
    return np.array(f)


def modified_t_score(X, y):
    """
    Calculate the Modified T-score for each feature.
    
    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input samples.
    y : numpy array, shape (n_samples, )
        The classes for the samples. There can be only 2 classes.

    Returns
    -------
    Score for each feature as a numpy array, shape (n_features, ). The higher the better.

    See Also
    --------

    For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.

    Examples
    --------

    >>> import sklearn.datasets as datasets
    >>> import numpy as np
    >>> from ITMO_FS.filters.univariate import modified_t_score
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 1, 2, 1, 2])
    >>> scores = modified_t_score(X, y)
    >>> print(scores)
    [1.68968099 0.12148022 0.39653932 0.17682997 2.04387142]

    """

    classes = np.unique(y)

    size_class0 = y[y == classes[0]].size
    size_class1 = y[y == classes[1]].size

    mean_class0 = np.mean(X[y == classes[0]], axis=0)
    mean_class0 = np.nan_to_num(mean_class0)
    mean_class1 = np.mean(X[y == classes[1]], axis=0)
    mean_class1 = np.nan_to_num(mean_class1)

    std_class0 = np.std(X[y == classes[0]], axis=0)
    std_class0 = np.nan_to_num(std_class0)
    std_class1 = np.std(X[y == classes[1]], axis=0)
    std_class1 = np.nan_to_num(std_class1)

    corr_with_y = np.apply_along_axis(lambda feature: abs(np.corrcoef(feature, y)[0][1]), 0, X)
    corr_with_y = np.nan_to_num(corr_with_y)

    corr_with_others = abs(np.corrcoef(X, rowvar=False))
    corr_with_others = np.nan_to_num(corr_with_others)

    mean_of_corr_with_others = (corr_with_others.sum(axis=1) - corr_with_others.diagonal()) / (
            len(corr_with_others) - 1)

    t_score_numerator = abs(mean_class0 - mean_class1)
    t_score_denominator = np.sqrt(
        (size_class0 * np.square(std_class0) + size_class1 * np.square(std_class1)) / (size_class0 + size_class1))
    modificator = corr_with_y / mean_of_corr_with_others

    modified_t_score = t_score_numerator / t_score_denominator * modificator
    modified_t_score = np.nan_to_num(modified_t_score)

    return modified_t_score


GLOB_MEASURE = {"FitCriterion": fit_criterion_measure,
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
                "ModifiedTScore": modified_t_score}


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
    if type(k) != int:
        raise TypeError("Number of features should be integer")
    if k > len(scores):
        raise ValueError("Cannot select %d features with n_features = %d" % (k, len(scores)))
    return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


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


GLOB_CR = {"Best by value": select_best_by_value,
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
