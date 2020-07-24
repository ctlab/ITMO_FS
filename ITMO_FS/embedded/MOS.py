import random as rnd

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier

from ITMO_FS.utils import augmented_rvalue


class MOS(object):
    """
        Performs Minimizing Overlapping Selection under SMOTE (MOSS) or under No-Sampling (MOSNS) algorithm.

        Parameters
        ----------
        model : constructor
            The constructor of the model that will be used. Currently only SGDClassifier should be passed, 
            other models would not work.
        loss : str, 'log' or 'hinge'
            Loss function to use in the algorithm. 'log' gives a logistic regression, while 'hinge'
            gives a support vector machine. 
        seed : int
            Seed for python random.

        See Also
        --------
        https://www.sciencedirect.com/science/article/pii/S0169743919306070

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> dataset = make_classification(n_samples=100, n_features=20)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> for i in range(50):  # create imbalance between classes
        ...     target[i] = 0
        >>> print(MOS().runMOSS(data, target))
    """

    def __init__(self, model=SGDClassifier, loss='log',
                 seed=42):  # TODO Add wrapper function which will take weights from module
        if loss not in ['hinge', 'log']:
            raise KeyError("Loss should be 'hinge' or 'log', %r was passed" % loss)
        self.model = model
        self.loss = loss
        rnd.seed = seed
        self.selected_features = None

    def runMOSS(self, X, y, l1_ratio=0.5, threshold=10e-4, epochs=1000, alphas=np.arange(0.0002, 0.02, 0.0002)):
        """
        Runs the MOSS algorithm on the specified dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            The input samples.
        y : array-like, shape (n_samples)
            The classes for the samples.
        l1_ratio : float
            The value used to balance the L1 and L2 penalties in elastic-net.
        threshold : float
            The threshold value for feature dropout. Instead of comparing them to zero, they are normalized
            and values with absolute value lower than the threshold are dropped out.
        epochs : int
            The number of epochs to perform in logistic regression.
        alphas : array-like, shape (n_alphas)
            The range of lambdas that should form the regularization path.
        Returns
        ------
        array-like, shape (n_samples,n_selected_features) : the resulting dataset with remaining features
        """

        newX, newY = SMOTE(random_state=rnd.seed).fit_resample(X, y)
        return self.runMOSNS(newX, newY, l1_ratio, threshold, epochs, alphas)

    def runMOSNS(self, X, y, l1_ratio=0.5, threshold=10e-4, epochs=1000, alphas=np.arange(0.0002, 0.02, 0.0002)):
        """
        Runs the MOSNS algorithm on the specified dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples,n_features)
            The input samples.
        y : array-like, shape (n_samples)
            The classes for the samples.
        l1_ratio : float
            The value used to balance the L1 and L2 penalties in elastic-net.
        threshold : float
            The threshold value for feature dropout. Instead of comparing them to zero, they are normalized 
            and values with absolute value lower than the threshold are dropped out.
        epochs : int
            The number of epochs to perform in gradient descent.
        alphas : array-like, shape (n_alphas)
            The range of lambdas that should form the regularization path.
        Returns
        ------
        array-like, shape (n_samples,n_selected_features) : the resulting dataset with remaining features
        """
        min_rvalue = 1
        min_b = []
        for a in alphas:  # TODO: do a little more research on the range of lambdas
            model = self.model(loss=self.loss, random_state=rnd.seed, penalty='elasticnet',
                               alpha=a, l1_ratio=l1_ratio, max_iter=epochs)
            model.fit(X, y)
            b = model.coef_[0]
            rvalue = augmented_rvalue(X[:, [i for i in range(X.shape[1]) if np.abs(b[i]) > threshold]], y)
            if min_rvalue > rvalue:
                min_rvalue = rvalue
                min_b = b
        self.selected_features = [i for i in range(X.shape[1]) if np.abs(min_b[i]) > threshold]
        return X[:, self.selected_features]
