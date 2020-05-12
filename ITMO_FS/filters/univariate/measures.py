from functools import partial
from math import exp
from math import log

import numpy as np
from scipy import sparse as sp
from scipy import special

from ITMO_FS.utils.data_check import generate_features
from ITMO_FS.utils.information_theory import conditional_entropy
from ITMO_FS.utils.information_theory import entropy
from ITMO_FS.utils.qpfs_body import qpfs_body

from scipy.sparse import csc_matrix, lil_matrix


def fit_criterion_measure(X, y):
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
        inter_class += n * np.power((mu - mean_feature), 2)  # TODO: something went horribly wrong here
        intra_class += (n - 1) * var
    f_ratio = inter_class / intra_class
    return f_ratio


def f_ratio_measure(X, y):
    #TODO devision by zero
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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import f_ratio_measure

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = f_ratio_measure(X, y)
    print(scores)
    """
    return np.apply_along_axis(__calculate_F_ratio, 0, X, y)


def gini_index(X, y):
    """
    Gini index is a measure of statistical dispersion.

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
    """
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
    https://www.matec-conferences.org/articles/matecconf/pdf/2016/05/matecconf_iccma2016_06002.pdf

    Examples
    --------
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import su_measure

    X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
    y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    scores = su_measure(X, y)
    print(scores)
    """
    entropy_y = entropy(y)
    f_ratios = np.empty(X.shape[1])
    for index in range(X.shape[1]):
        entropy_x = entropy(X[:, index])
        cond_entropy = conditional_entropy(X[:, index], y)
        f_ratios[index] = 2 * (entropy_y - cond_entropy) / (entropy_x + entropy_y)
    return f_ratios


# TODO concordation coef, kendal coef

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
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import fechner_corr

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = fechner_corr(X, y)
    print(scores)
    """
    y_mean = np.mean(y)
    if len(X.shape) == 1:
        m = 1
        n = X.shape[0]
    else:
        n, m = X.shape
    y_dev = y - y_mean
    if m == 1:
        x_col_mean = np.mean(X)
    else:
        x_col_mean = np.mean(X, axis=0)
    x_dev = X - x_col_mean
    if m == 1:
        # TODO fix m == 1 case (The sum tries to go over 0 columns raising an error.
        #  It needs to be transformed to a two-dimensional array)
        f_ratios = np.array(
            [np.sum((x_dev >= 0).T & (y_dev >= 0), axis=1) + np.sum((x_dev <= 0).T & (y_dev <= 0), axis=1)]).astype(
            float)
    else:
        f_ratios = np.sum((x_dev >= 0).T & (y_dev >= 0), axis=1) + np.sum((x_dev <= 0).T & (y_dev <= 0), axis=1).astype(
            float)
    # TODO Count (Na-Nb)/N, for now Na/N is counted (possible fix: f_ratios = -1 + 2*f_ratios/n after simplification)
    f_ratios /= n
    return f_ratios


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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import realiefF_measure

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = reliefF_measure(X, y)
    print(scores)

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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import spearman_corr

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = spearman_corr(X, y)
    print(scores)

    """
    n = X.shape[0]
    c = 6 / (n * (n - 1) * (n + 1))
    # TODO It must count differences of ranks, not of values
    if y.shape == X.shape:
        dif = X - y
    else:
        dif = X - np.repeat(y, X.shape[1]).reshape(y.shape[0], X.shape[1])
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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import pearson_corr

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = pearson_corr(X, y)
    print(scores)

    """
    x_dev = X - np.mean(X, axis=0)
    y_dev = y - np.mean(y, axis=0)
    sum_dev = y_dev.T.dot(x_dev)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    denominators = np.sqrt(np.sum(sq_dev_y, axis=0) * np.sum(sq_dev_x, axis=0))
    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i in range(len(denominators))])
    return results


# TODO need to implement unsupervised way
# TODO add sparse functionality
def laplacian_score(X, y, k_neighbors=5, t=1,
                    metric=np.linalg.norm, **kwargs):
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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import laplacian_score

    data = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    X = np.array(data[0])
    y = np.array(data[1])
    scores = laplacian_score(X, y)
    features = sorted(range(len(scores)), key = lambda k: scores[k])
    print(features)

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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import information_gain

    X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
    y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
    scores = information_gain(X, y)
    print(scores)

    """
    entropy_x = np.apply_along_axis(entropy, 0, X)
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
    import sklearn.datasets as datasets
    from ITMO_FS.filters.univariate import anova

    X, y = datasets.make_classification(n_samples=200, n_features=7, shuffle=False)
    scores = anova(X, y)
    print(scores)

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


GLOB_MEASURE = {"FitCriterion": fit_criterion_measure,
                "FRatio": f_ratio_measure,
                "GiniIndex": gini_index,
                "SymmetricUncertainty": su_measure,
                "SpearmanCorr": spearman_corr,
                "PearsonCorr": pearson_corr,
                "FechnerCorr": fechner_corr,
                "ReliefF": reliefF_measure,
                "Chi2": chi2_measure,
                "InformationGain": information_gain}


def select_best_by_value(value):
    return partial(__select_by_value, value=value, more=True)


def select_worst_by_value(value):
    return partial(__select_by_value, value=value, more=False)


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
    return partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if type(k) != int:
        raise TypeError("Number of features should be integer")
    return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


GLOB_CR = {"Best by value": select_best_by_value,
           "Worst by value": select_worst_by_value,
           "K best": select_k_best,
           "K worst": select_k_worst}


def qpfs_filter(X, y, r=None, sigma=None, solv='quadprog', fn=pearson_corr):
    """
    Performs Quadratic Programming Feature Selection algorithm.
    Note that this realization requires labels to start from 1 and be numberical.
    
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
    
    from ITMO_FS.filters.univariate import qpfs_filter
    from sklearn.datasets import make_classification

    x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    y = np.array([1, 3, 2, 1, 2])
    ranks = qpfs_filter(x, y)
    print(ranks)

    """

    return qpfs_body(X, y, fn, r=r, sigma=sigma, solv=solv)
