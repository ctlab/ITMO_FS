from ...utils.information_theory import *


def MIM(selected_features, free_features, X, y):
    """
        Mutual Information Maximization feature scoring criterion.
        This criterion focuses only on increase of relevance.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import MIM
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(MIM(np.array(selected_features), np.array(other_features), data, target))

    """
    return matrix_mutual_information(X[:, free_features], y)


def MRMR(selected_features, free_features, X, y):
    """
        Minimum-Redundancy Maximum-Relevance feature scoring criterion.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import MRMR
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(MRMR(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    return generalizedCriteria(selected_features, free_features, X, y, 1 / selected_features.size, 0)


def JMI(selected_features, free_features, X, y):
    """
        Joint Mutual Information feature scoring criterion.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import JMI
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(JMI(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    return generalizedCriteria(selected_features, free_features, X, y, 1 / selected_features.size,
                               1 / selected_features.size)


def CIFE(selected_features, free_features, X, y):
    """
        Conditional Infomax Feature Extraction feature scoring criterion.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import CIFE
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(CIFE(np.array(selected_features), np.array(other_features), data, target))

    """
    return generalizedCriteria(selected_features, free_features, X, y, 1, 1)


def MIFS(selected_features, free_features, X, y, beta):
    """
        Mutual Information Feature Selection feature scoring criterion.
        This criterion includes the I(X;Y) term to ensure feature relevance, but introduces a 
        penalty to enforce low correlations with features already selected in set.
        Given set of already selected features and set of remaining features on dataset X
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
        beta : float,
            coeficient for redundancy term
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import MIFS
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(MIFS(np.array(selected_features), np.array(other_features), data, target, 0.4))

    """
    return generalizedCriteria(selected_features, free_features, X, y, beta, 0)


def CMIM(selected_features, free_features, X, y):
    """
        Conditional Mutual Info Maximisation feature scoring criterion.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import CMIM
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(CMIM(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    vectorized_function = lambda free_feature: \
        min(np.vectorize(
            lambda selected_feature: conditional_mutual_information(X[:, free_feature], y, X[:, selected_feature]))(
            selected_features))
    return np.vectorize(vectorized_function)(free_features)


def ICAP(selected_features, free_features, X, y):
    """
        Interaction Capping feature scoring criterion.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------

        http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import ICAP
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(ICAP(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    relevance = matrix_mutual_information(X[:, free_features], y)
    redundancy = np.vectorize(
        lambda free_feature: np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(
        free_features)
    cond_dependency = np.vectorize(lambda free_feature: \
                                       np.sum(np.apply_along_axis(conditional_mutual_information, 0,
                                                                  X[:, selected_features], X[:, free_feature], y)))(
        free_features)
    return relevance - np.maximum(redundancy - cond_dependency, 0.)


def DCSF(selected_features, free_features, X, y):
    """
        Dynamic change of selected feature with the class scoring criterion.
        DCSF employs both mutual information and conditional mutual information 
        to find an optimal subset of features.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        
        https://www.sciencedirect.com/science/article/abs/pii/S0031320318300736
        
        Examples
        --------
        
        from ITMO_FS.filters.multivariate import DCSF
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(DCSF(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(lambda z, x, y: conditional_mutual_information(x, y, z), 0, X[:, selected_features],
                            X[:, free_feature], y) +
        np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], y, X[:, free_feature]) -
        matrix_mutual_information(X[:, selected_features], X[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def CFR(selected_features, free_features, X, y):
    """
        The criterion of CFR maximizes the correlation and minimizes the redundancy.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        
        https://www.sciencedirect.com/science/article/pii/S2210832719302522
        
        Examples
        --------
        
        from ITMO_FS.filters.multivariate import CFR
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(CFR(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(lambda z, x, y: conditional_mutual_information(x, y, z), 0, X[:, selected_features],
                            X[:, free_feature], y) +
        np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y) -
        matrix_mutual_information(X[:, selected_features], X[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def MRI(selected_features, free_features, X, y):
    """
        Max-Relevance and Max-Independence feature scoring criteria.
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        
        https://link.springer.com/article/10.1007/s10489-019-01597-z
        
        Examples
        --------
        
        from ITMO_FS.filters.multivariate import MRI
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(MRI(np.array(selected_features), np.array(other_features), data, target))

    """
    return generalizedCriteria(selected_features, free_features, X, y, 2 / (selected_features.size + 1),
                               2 / (selected_features.size + 1))


def __information_weight(Xk, Xj, y):
    return 1 + (joint_mutual_information(Xk, Xj, y) + mutual_information(Xk, y) + mutual_information(Xj, y)) / (
            entropy(Xk) + entropy(Xj))


def __SU(Xk, Xj):
    return 2 * mutual_information(Xk, Xj) / (entropy(Xk) + entropy(Xj))


def IWFS(selected_features, free_features, X, y):
    """
        Interaction Weight base feature scoring criteria.
        IWFS is good at identifyng
        Given set of already selected features and set of remaining features on dataset X
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
        
        See Also
        --------
        
        https://www.sciencedirect.com/science/article/abs/pii/S0031320315000850

        Examples
        --------
        
        from ITMO_FS.filters.multivariate import IWFS
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(IWFS(np.array(selected_features), np.array(other_features), data, target))

    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.prod(
        np.apply_along_axis(lambda Xj, Xk, y: __information_weight(Xk, Xj, y), 0, X[:, selected_features],
                            X[:, free_feature], y) *
        (np.apply_along_axis(__SU, 0, X[:, selected_features], X[:, free_feature]) + 1))
    return np.vectorize(vectorized_function)(free_features)


# Ask question what should happen if number of features user want is less than useful number of features
def generalizedCriteria(selected_features, free_features, X, y, beta, gamma):
    """
        This feature scoring criteria is a linear combination of all relevance,
        redundancy, conditional depenedency
        Given set of already selected features and set of remaining features on dataset X
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
        beta : float,
            coeficient for redundancy term
        gamma : float,
            coeficient for conditional dependancy term    
        
        See Also
        --------

        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
        
        Examples
        --------
        
        from ITMO_FS.filters.multivariate import CFR
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        selected_features = [1, 2]
        other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
        print(generalizedCriteria(np.array(selected_features), np.array(other_features), data, target, 0.4, 0.3))

    """
    if selected_features.size == 0:
        return matrix_mutual_information(X, y)
    relevance = np.apply_along_axis(mutual_information, 0, X[:, free_features], y)
    redundancy = np.vectorize(
        lambda free_feature: np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(
        free_features)

    cond_dependency = np.vectorize(lambda free_feature: np.sum(np.apply_along_axis(conditional_mutual_information, 0,
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
