from ...utils.information_theory import *


def MIM(selected_features, free_features, x, y):
    """
    Mutual Information Maximization feature scoring criterion. This
    criterion focuses only on increase of relevance. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.


        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import MIM
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> MIM(np.array(selected_features), np.array(other_features), x, y)
        array([1.33217904, 0.67301167, 1.60943791])
    """
    return matrix_mutual_information(x[:, free_features], y)


def MRMR(selected_features, free_features, x, y):
    """
    Minimum-Redundancy Maximum-Relevance feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.


        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import MRMR
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> MRMR(np.array(selected_features), np.array(other_features), x, y)
        array([0.80471896, 0.33650583, 0.94334839])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    return generalizedCriteria(selected_features, free_features, x, y,
                               1 / selected_features.size, 0)


def JMI(selected_features, free_features, x, y):
    """
    Joint Mutual Information feature scoring criterion. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.


        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import JMI
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> JMI(np.array(selected_features), np.array(other_features), x, y)
        array([0.80471896, 0.33650583, 0.94334839])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    return generalizedCriteria(selected_features, free_features, x, y,
                               1 / selected_features.size,
                               1 / selected_features.size)


def CIFE(selected_features, free_features, x, y):
    """
    Conditional Infomax Feature Extraction feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
                Notes ----- For more details see `this paper
                <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf
                />`_.

        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import CIFE
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> CIFE(np.array(selected_features), np.array(other_features), x, y)
        array([0.27725887, 0.        , 0.27725887])
    """
    return generalizedCriteria(selected_features, free_features, x, y, 1, 1)


def MIFS(selected_features, free_features, x, y, beta):
    """
    Mutual Information Feature Selection feature scoring criterion. This
    criterion includes the I(X;Y) term to ensure feature relevance,
    but introduces a penalty to enforce low correlations with features
    already selected in set. Given set of already selected features and set
    of remaining features on dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        beta : float,
            coeficient for redundancy term
        
                Notes ----- For more details see `this paper
                <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf
                />`_.

        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import MIFS
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> MIFS(np.array(selected_features), np.array(other_features), x, y, 0.4)
        array([0.91021097, 0.403807  , 1.0765663 ])
    """
    return generalizedCriteria(selected_features, free_features, x, y, beta, 0)


def CMIM(selected_features, free_features, x, y):
    """
    Conditional Mutual Info Maximisation feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.


        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import CMIM
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> CMIM(np.array(selected_features), np.array(other_features), x, y)
        array([0.27725887, 0.        , 0.27725887])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    vectorized_function = lambda free_feature: \
        min(np.vectorize(
            lambda selected_feature: conditional_mutual_information(
                x[:, free_feature], y, x[:, selected_feature]))(
            selected_features))
    return np.vectorize(vectorized_function)(free_features)


def ICAP(selected_features, free_features, x, y):
    """
    Interaction Capping feature scoring criterion. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.



        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import ICAP
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> ICAP(np.array(selected_features), np.array(other_features), x, y)
        array([0.27725887, 0.        , 0.27725887])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    relevance = matrix_mutual_information(x[:, free_features], y)
    redundancy = np.vectorize(
        lambda free_feature: np.sum(
            matrix_mutual_information(x[:, selected_features],
                                      x[:, free_feature])))(
        free_features)
    cond_dependency = (np.vectorize(lambda free_feature:
                                    np.sum(np.apply_along_axis(
                                        conditional_mutual_information, 0,
                                        x[:, selected_features],
                                        x[:, free_feature], y)))(
        free_features))
    return relevance - np.maximum(redundancy - cond_dependency, 0.)


def DCSF(selected_features, free_features, x, y):
    """
    Dynamic change of selected feature with the class scoring criterion.
    DCSF employs both mutual information and conditional mutual information
    to find an optimal subset of features. Given set of already selected
    features and set of remaining features on dataset X with labels y
    selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <https://www.sciencedirect.com/science/article/abs/pii
        /S0031320318300736/>`_.
        

        
        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import DCSF
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> DCSF(np.array(selected_features), np.array(other_features), x, y)
        array([0.83177662, 0.65916737, 0.55451774])
    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(
            lambda z, a, b: conditional_mutual_information(a, b, z), 0,
            x[:, selected_features],
            x[:, free_feature], y) +
        np.apply_along_axis(conditional_mutual_information, 0,
                            x[:, selected_features], y, x[:, free_feature]) -
        matrix_mutual_information(x[:, selected_features], x[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def CFR(selected_features, free_features, x, y):
    """
    The criterion of CFR maximizes the correlation and minimizes the
    redundancy. Given set of already selected features and set of remaining
    features on dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <https://www.sciencedirect.com/science/article/pii/S2210832719302522
        />`_.

        
        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import CFR
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(x)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, x.shape[1]) if i not in selected_features]
        >>> CFR(np.array(selected_features), np.array(other_features), x, y)
        array([0.55451774, 0.        , 0.55451774])

    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(
            lambda z, a, b: conditional_mutual_information(a, b, z), 0,
            x[:, selected_features],
            x[:, free_feature], y) +
        np.apply_along_axis(conditional_mutual_information, 0,
                            x[:, selected_features], x[:, free_feature], y) -
        matrix_mutual_information(x[:, selected_features], x[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def MRI(selected_features, free_features, X, y):
    """
    Max-Relevance and Max-Independence feature scoring criteria. Given set
    of already selected features and set of remaining features on dataset X
    with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <https://link.springer.com/article/10.1007/s10489-019-01597-z/>`_.

        
        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import MRI
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(X)
        >>> selected_features = [1, 2]

        >>> other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
        >>> MRI(np.array(selected_features), np.array(other_features), X, y)
        array([0.62889893, 0.22433722, 0.72131855])
    """
    return generalizedCriteria(selected_features, free_features, X, y,
                               2 / (selected_features.size + 1),
                               2 / (selected_features.size + 1))


def __information_weight(xk, xj, y):
    return (1 +
            (joint_mutual_information(xk, xj, y) +
             mutual_information(xk, y) +
             mutual_information(xj, y)) / (entropy(xk) + entropy(xj)))


def __SU(xk, xj):
    return 2 * mutual_information(xk, xj) / (entropy(xk) + entropy(xj))


def IWFS(selected_features, free_features, X, y):
    """
    Interaction Weight base feature scoring criteria. IWFS is good at
    identifyng Given set of already selected features and set of remaining
    features on dataset X with labels y selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        
        Notes ----- For more details see `this paper
        <https://www.sciencedirect.com/science/article/abs/pii
        /S0031320315000850/>`_.
        


        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import IWFS
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(X)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
        >>> IWFS(np.array(selected_features), np.array(other_features), X, y)
        array([13.9984669 , 13.35861706, 14.56253175])
    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.prod(
        np.apply_along_axis(lambda Xj, Xk, y: __information_weight(Xk, Xj, y),
                            0, X[:, selected_features],
                            X[:, free_feature], y) *
        (np.apply_along_axis(__SU, 0, X[:, selected_features],
                             X[:, free_feature]) + 1))
    return np.vectorize(vectorized_function)(free_features)


# Ask question what should happen if number of features user want is less
# than useful number of features
def generalizedCriteria(selected_features, free_features, X, y, beta, gamma):
    """
    This feature scoring criteria is a linear combination of all relevance,
    redundancy, conditional dependency Given set of already selected
    features and set of remaining features on dataset X with labels y
    selects next feature.

        Parameters
        ----------
        selected_features : list of ints,
            already selected features
        free_features : list of ints
            free features
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.
        beta : float,
            coeficient for redundancy term
        gamma : float,
            coeficient for conditional dependancy term

        Notes
        -----
        See the original paper [1]_ for more details.

        References ---------- .. [1] Brown, Gavin et al. "Conditional
        Likelihood Maximisation: A Unifying Framework for Information
        Theoretic Feature Selection." JMLR 2012.

        Examples
        --------

        >>> from ITMO_FS.filters.multivariate import CFR
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> est.fit()
        KBinsDiscretizer(encode='ordinal', n_bins=10)
        >>> X = est.transform(X)
        >>> selected_features = [1, 2]
        >>> other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
        >>> generalizedCriteria(np.array(selected_features), np.array(other_features), X, y, 0.4, 0.3)
        array([0.91021097, 0.403807  , 1.0765663 ])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    relevance = np.apply_along_axis(mutual_information, 0, X[:, free_features],
                                    y)
    redundancy = np.vectorize(
        lambda free_feature: np.sum(
            matrix_mutual_information(X[:, selected_features],
                                      X[:, free_feature])))(
        free_features)

    cond_dependency = np.vectorize(lambda free_feature: np.sum(
        np.apply_along_axis(conditional_mutual_information, 0,
                            X[:, selected_features],
                            X[:, free_feature], y)))(
        free_features)
    return relevance - beta * redundancy + gamma * cond_dependency


GLOB_MEASURE = {"MIM": MIM,
                "MRMR": MRMR,
                "JMI": JMI,
                "CIFE": CIFE,
                "MIFS": MIFS,
                "CMIM": CMIM,
                "ICAP": ICAP,
                "DCSF": DCSF,
                "CFR": CFR,
                "MRI": MRI,
                "IWFS": IWFS,
                "generalizedCriteria": generalizedCriteria}
