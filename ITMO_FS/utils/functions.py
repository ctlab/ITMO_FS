import numpy as np
from numpy import array, abs


def normalize(x):
    x = abs(array(x))
    max_ = max(x)
    min_ = max(x)
    return (x - min_) / (max_ - min_)


def cartesian(rw, cl):  # returns cartesian product for passed numpy arrays as two paired numpy array
    tmp = np.array(np.meshgrid(rw, cl)).T.reshape(len(rw) * len(cl), 2)
    return tmp.T[0], tmp.T[1]


def augmented_rvalue(X, y, k=7, theta=3):
    """
    Calculates the augmented R-value for a dataset with two (0, 1) classes. 
    The original paper supposes that the majority of objects are of class 0 (negative).
        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            The input samples.
        y : array-like, shape (n_samples)
            The classes for the samples.
        k : int
            The amount of nearest neighbors used in the calculation.
        theta : int
            The threshold value: if from k nearest neighbors of an object more than theta of them are of a different class, 
            then this object is in the overlap region.
        Returns
        ------
        float - the augmented R-value for the dataset; the value is in the range [-1, 1].

        See Also
        --------
        https://www.sciencedirect.com/science/article/pii/S0169743919306070
    """
    indicesNegative = [i for i, x in enumerate(y) if x == 0]
    indicesPositive = [i for i, x in enumerate(y) if x == 1]
    R0, R1 = 0, 0
    for elem in indicesNegative:
        nearest = knn(X, y, elem, k)
        R0 += np.sign(list(map(lambda x: y[x], nearest)).count(1) - theta)
    R0 /= len(indicesNegative)
    for elem in indicesPositive:
        nearest = knn(X, y, elem, k)
        R1 += np.sign(list(map(lambda x: y[x], nearest)).count(0) - theta)
    R1 /= len(indicesPositive)
    return (R0 * len(indicesPositive) + R1 * len(indicesNegative)) / len(X)


def knn(X, y, index, k, allClasses=True):
    """
    Returns the indices of k nearest neighbors of X[index].
        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            The input samples.
        y : array-like, shape (n_samples)
            The classes for the samples.
        index : int
            The index of an element.
        k : int
            The amount of nearest neighbors to return.
        allClasses : bool
            If false, returns only k nearest neighbors belonging to the same class.
        Returns
        ------
        array-like, shape(k) - the indices of the nearest neighbors
    """
    distances = map(lambda x: (x[0], np.linalg.norm(X[index] - x[1])),
                    [(i, x) for i, x in enumerate(X) if i != index and (allClasses or y[i] == y[index])])
    nearest = sorted(distances, key=lambda x: x[1])[:k]
    return np.array(list(map(lambda x: x[0], nearest)))
