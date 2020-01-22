import numpy as np
import random as rnd

from .filter_utils import augmented_rvalue, SMOTE

class MOSS(object):
	"""
        Performs Minimizing Overlapping Selection under SMOTE (MOSS) algorithm.
        Note that this realization requires (-1, 1) labels instead of (0, 1).
        Parameters
        ----------
        seed : int
            Seed for python random

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
		print(MOSS().run(data, target))
    """
	def __init__(self, seed=42):
		rnd.seed = seed

	def run(self, X, y, alpha=0.5, nu=0.0001, threshold=10e-4, epochs=10000):
		"""
        Runs the MOSS algorithm on the specified dataset.
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
        array-like : the indices of features to leave
        """

		newX, newY = SMOTE(X, y)
		min_rvalue = 1
		min_b = []
		for l in np.arange(0.0, 1.0, 0.05):  # TODO: do a little more research on the range of lambdas; random datasets work well with 0-1
			b = np.array([rnd.random() / 10 for i in range(newX.shape[1])])
			b0 = rnd.random() / 10
			for epoch in range(epochs):
				oldB = b
				for i in range(X.shape[0]):  # stochastic gradient descent doesn't really work too well here, full/batch should be better
					e = np.exp(- newY[i] * np.dot(oldB, newX[i]))
					b += (nu / newX.shape[0]) * (e * newY[i] * newX[i]) / np.log(1 + e) 
				b -= nu * (l * alpha * np.sign(oldB) + l * (1 - alpha) * oldB)
			b /= np.max(b)
			rvalue = augmented_rvalue(newX[:, [i for i in range(newX.shape[1]) if np.abs(b[i]) > threshold]], newY)
			if min_rvalue > rvalue:
				min_rvalue = rvalue
				min_b = b
		return np.array([i for i in range(newX.shape[1]) if np.abs(min_b[i]) > threshold])
        