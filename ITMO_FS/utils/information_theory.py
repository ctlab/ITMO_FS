from collections import Counter
from itertools import groupby
from math import log, fsum
from operator import itemgetter

import numpy as np


def conditional_entropy(x_j, y):
    """Calculate the conditional entropy (H(Y|X)) between two arrays.

    Parameters
    ----------
    x_j : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    float : H(Y|X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import conditional_entropy
    >>> conditional_entropy([1,2,1,3,4], [1,2,3,4,5])
    0.2772588722239781
    >>> conditional_entropy([1], [2])
    0.0
    >>> conditional_entropy([1,2,1,3,2,4], [3,3,3,3,3,3])
    0.0
    >>> conditional_entropy([1,2,3,1,3,2,3,4,1], [1,2,1,3,1,4,4,1,5])
    0.7324081924454064
    """
    buf = [[e[1] for e in g] for _, g in 
           groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x_j)


def joint_entropy(*arrs):
    """Calculate the joint entropy (H(X;Y;...)) between multiple arrays.

    Parameters
    ----------
    arrs : any number of array-like, all of shape (n,)
        Any number of arrays.

    Returns
    -------
    float : H(X;Y;...) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import joint_entropy
    >>> joint_entropy([1,2,3,4,5])
    1.6094379124341003
    >>> joint_entropy([1,2,3,4,4])
    1.3321790402101221
    >>> joint_entropy([1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6])
    1.791759469228055
    >>> joint_entropy([1,2,1,3,2], [3,3,3,3,3])
    1.0549201679861442
    >>> conditional_entropy([1,1], [2,2])
    0.0
    """
    return entropy(list(zip(*arrs)))


def matrix_mutual_information(x, y):
    """Calculate the mutual information (I(X;Y) = H(Y) - H(Y|X)) between each
    column of the matrix and an array.

    Parameters
    ----------
    x : array-like, shape (n, n_features)
        The matrix.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    array-like, shape (n_features,) : I(X;Y) values for all columns of the
    matrix

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import matrix_mutual_information
    >>> matrix_mutual_information([[1,3,2,1], [2,2,2,1], [3,3,2,2]], [1,1,2])
    array([0.63651417, 0.17441605, 0.        , 0.63651417])
    """
    return np.apply_along_axis(mutual_information, 0, x, y)


def mutual_information(x, y):
    """Calculate the mutual information (I(X;Y) = H(Y) - H(Y|X)) between two
    arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    float : I(X;Y) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import mutual_information
    >>> mutual_information([1,2,3,4,5], [5,4,3,2,1])
    1.6094379124341003
    >>> mutual_information([1,2,3,1,2,3,1,2,3], [1,1,2,2,3,3,4,4,5])
    0.48248146150371407
    >>> mutual_information([1,2,3], [1,1,1])
    0.0
    >>> mutual_information([1,2,1,3,2,4,3,1], [1,2,3,4,2,3,2,1])
    0.9089087348987808
    """
    return entropy(y) - conditional_entropy(x, y)


def conditional_mutual_information(x, y, z):
    """Calculate the conditional mutual information (I(X;Y|Z) = H(X;Z) + H(Y;Z)
    - H(X;Y;Z) - H(Z)) between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X;Y|Z) value

    Examples
    --------
    >>> from ITMO_FS.utils import conditional_mutual_information
    >>> conditional_mutual_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.3465735902799726
    >>> conditional_mutual_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.0
    >>> conditional_mutual_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    1.054920167986144
    >>> conditional_mutual_information([1,2,3], [1,1,1], [3,2,2])
    0.0
    >>> conditional_mutual_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.27725887222397816
    """
    return (entropy(list(zip(x, z)))
            + entropy(list(zip(y, z)))
            - entropy(list(zip(x, y, z)))
            - entropy(z))


def joint_mutual_information(x, y, z):
    """Calculate the joint mutual information (I(X,Y;Z) = I(X;Z) + I(Y;Z|X))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X,Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import joint_mutual_information
    >>> joint_mutual_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.6931471805599454
    >>> joint_mutual_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.39575279478527814
    >>> joint_mutual_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> joint_mutual_information([1,2,3], [1,1,1], [3,2,2])
    0.636514168294813
    >>> joint_mutual_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    1.5571130980576458
    """
    return mutual_information(x, z) + conditional_mutual_information(y, z, x)


def interaction_information(x, y, z):
    """Calculate the interaction information (I(X;Y;Z) = I(X;Y) - I(X;Y|Z))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X;Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import interaction_information
    >>> interaction_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    -0.13081203594113694
    >>> interaction_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.0
    >>> interaction_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> interaction_information([1,2,3], [1,1,1], [3,2,2])
    0.0
    >>> interaction_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.6730116670092565
    """
    return mutual_information(x, y) - conditional_mutual_information(x, y, z)


def symmetrical_relevance(x, y, z):
    """Calculate the symmetrical relevance (SR(X;Y;Z) = I(X;Y;Z) / H(X;Y|Z))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : SR(X;Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import symmetrical_relevance
    >>> symmetrical_relevance([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.5000000000000001
    >>> symmetrical_relevance([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.2458950368496943
    >>> symmetrical_relevance([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> symmetrical_relevance([1,2,3], [1,1,1], [3,2,2])
    0.5793801642856952
    >>> symmetrical_relevance([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.6762456261857126
    """
    return joint_mutual_information(x, y, z) / joint_entropy(x, y, z)

def entropy(x):
    """Calculate the entropy (H(X)) of an array.

    Parameters
    ----------
    x : array-like, shape (n,)
        The array.

    Returns
    -------
    float : H(X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import entropy
    >>> entropy([1,1,1])
    0.0
    >>> entropy([1,2,3,4,5])
    1.6094379124341003
    >>> entropy([5,4,1,2,3])
    1.6094379124341003
    >>> entropy([1,2,1,2,1,2,1,2,1,2])
    0.6931471805599456
    >>> entropy([1,1,1,1,1,2])
    0.4505612088663047
    """
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)
