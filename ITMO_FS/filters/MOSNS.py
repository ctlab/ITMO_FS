import numpy as np
import random as rnd

from .filter_utils import augmented_rvalue

class MOSNS(object):
    """
        Performs Minimizing Overlapping Selection under No-Sampling (MOSNS) algorithm.
        Note that this realization requires (-1, 1) labels instead of (0, 1).
        Parameters
        ----------
        seed : int
            The seed for python random.

        See Also
        --------
        https://www.sciencedirect.com/science/article/pii/S0169743919306070

        Examples
        --------
        dataset = make_classification(n_samples=100, n_features=20)
        data, target = np.array(dataset[0]), np.array(dataset[1])
        target = np.where(target==0, -1, target)  # replace zeroes with negative ones
        for i in range(50):  # create imbalance between classes
            target[i] = -1
        print(MOSNS().run(data, target))
    """

    def __init__(self, seed=42):
        rnd.seed = seed

    def run(self, X, y, alpha=0.5, nu=0.0001, threshold=10e-4, epochs=10000):
        """
        Runs the MOSNS algorithm on the specified dataset.
        A different loss function is used here than in the paper due to using (-1, 1) labels.
        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            The input samples.
        y : array-like, shape (n_samples)
            The classes for the samples.
        alpha : float
            The value used to balance the L1 and L2 penalties in elastic-net.
        nu : float
            The learning rate.
        threshold : float
            The threshold value for feature dropout. Instead of comparing them to zero, they are normalized 
            and values with absolute value lower than the threshold are dropped out.
        epochs : int
            The number of epochs to perform in gradient descent.
        Returns
        ------
        array-like, shape (n_samples,n_selected_features) : the resulting dataset with remaining features
        """
        min_rvalue = 1
        min_b = []
        for l in np.arange(0.0, 3.0, 0.1):  # TODO: do a little more research on the range of lambdas
            b = np.array([rnd.random() / 10 for i in range(X.shape[1])])
            b0 = rnd.random() / 10
            for epoch in range(epochs):  
                # TODO: should redo this as a proper SVM, this is probably not working as intended: 
                # in random data, noise features usually have higher values of b while important features have close-to-zero values
                oldB = b
                oldB0 = b0
                for i in range(X.shape[0]):
                    if 1 - y[i] * (np.dot(oldB, X[i]) + oldB0) > 0:
                        b += nu * y[i] * X[i]
                        b0 += nu * y[i]
                b -= nu * (l * alpha * np.sign(oldB) + l * (1 - alpha) * oldB)
            b /= np.max(b)
            rvalue = augmented_rvalue(X[:, [i for i in range(X.shape[1]) if np.abs(b[i]) > threshold]], y)
            if min_rvalue > rvalue:
                min_rvalue = rvalue
                min_b = b
        return X[:, ([i for i in range(X.shape[1]) if np.abs(min_b[i]) > threshold])]
