import random as rnd

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier

from ..utils import augmented_rvalue, DataChecker, generate_features


class MOS(DataChecker):
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

        Notes
        -----
        For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0169743919306070/>`_.

        Examples
        --------
        >>> from ITMO_FS.embedded import MOS
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> dataset = make_classification(n_samples=100, n_features=20)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> for i in range(50):  # create imbalance between classes
        ...     target[i] = 0
        >>> print(MOS().fit_transform(data, target))
    """

    def __init__(self, model=SGDClassifier, loss='log',
                 seed=42):  # TODO Add wrapper function which will take weights from module
        if loss not in ['hinge', 'log']:
            raise KeyError("Loss should be 'hinge' or 'log', %r was passed" % loss)
        self.model = model
        self.loss = loss
        rnd.seed = seed
        self.selected_features = None

    def fit(self, X, y, l1_ratio=0.5, threshold=10e-4, epochs=1000, alphas=np.arange(0.0002, 0.02, 0.0002), sampling=True, feature_names=None):
        """
            Runs the MOS algorithm on the specified dataset.

            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The input samples.
            y : array-like, shape (n_samples)
                The classes for the samples.
            l1_ratio : float, optional
                The value used to balance the L1 and L2 penalties in elastic-net.
            threshold : float, optional
                The threshold value for feature dropout. Instead of comparing them to zero, they are normalized 
                and values with absolute value lower than the threshold are dropped out.
            epochs : int, optional
                The number of epochs to perform in the algorithm.
            alphas : array-like, shape (n_alphas), optional
                The range of lambdas that should form the regularization path.
            sampling : bool, optional
                Bool value that control whether MOSS (True) or MOSNS (False) should be executed.
            feature_names : list of strings, optional
                    In case you want to define feature names

            Returns
            ------
            None
        """

        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        self.feature_names = dict(zip(features, feature_names))
        if sampling:
            X, y = SMOTE(random_state=rnd.seed).fit_resample(X, y)
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
        self.selected_features = features[[i for i in range(X.shape[1]) if np.abs(min_b[i]) > threshold]]

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            ------
            Transformed 2D numpy array
        """

        if type(X) is np.ndarray:
            return X[:, self.selected_features.astype(int)]
        else:
            return X[self.selected_features]

    def fit_transform(self, X, y, l1_ratio=0.5, threshold=10e-4, epochs=1000, alphas=np.arange(0.0002, 0.02, 0.0002), sampling=True, feature_names=None):
        """
            Fits the algorithm and transforms given dataset X.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            l1_ratio : float, optional
                The value used to balance the L1 and L2 penalties in elastic-net.
            threshold : float, optional
                The threshold value for feature dropout. Instead of comparing them to zero, they are normalized 
                and values with absolute value lower than the threshold are dropped out.
            epochs : int, optional
                The number of epochs to perform in gradient descent.
            alphas : array-like, shape (n_alphas), optional
                The range of lambdas that should form the regularization path.
            sampling : bool, optional
                Bool value that control whether MOSS (True) or MOSNS (False) should be executed.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            -------
            X dataset sliced with features selected by the algorithm
        """

        self.fit(X, y, feature_names=feature_names)
        return self.transform(X)

