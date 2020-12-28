import random as rnd

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone

from ..utils import augmented_rvalue, BaseTransformer, generate_features


class MOS(BaseTransformer):
    """
        Performs Minimizing Overlapping Selection under SMOTE (MOSS) or under No-Sampling (MOSNS) algorithm.

        Parameters
        ----------
        model : object
            The model that will be used. Currently only SGDClassifier should be passed, 
            other models would not work.
        weight_func : callable
            The function to extract weights from the model.
        loss : str, 'log' or 'hinge', optional
            Loss function to use in the algorithm. 'log' gives a logistic regression, while 'hinge'
            gives a support vector machine. 
        seed : int, optional
            Seed for python random.
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
        k_neighbors : int, optional
            Amount of nearest neighbors to use in SMOTE if MOSS is used.       

        Notes
        -----
        For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0169743919306070/>`_.

        Examples
        --------
        >>> from ITMO_FS.embedded import MOS
        >>> from sklearn.linear_model import SGDClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> dataset = make_classification(n_samples=100, n_features=20)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> for i in range(50):  # create imbalance between classes
        ...     target[i] = 0
        >>> m = MOS(model=SGDClassifier(), weight_func=lambda model: model.coef_[0])
        >>> m.fit(data, target)
        >>> m.transform(data).shape[0]
        100
    """

    def __init__(self, model, weight_func, loss='log', seed=42, l1_ratio=0.5, threshold=10e-4, epochs=1000, 
                    alphas=np.arange(0.0002, 0.02, 0.0002), sampling=False, k_neighbors=2):
        self.model = model
        self.weight_func = weight_func
        self.loss = loss
        self.seed = seed
        self.l1_ratio = l1_ratio
        self.threshold = threshold
        self.epochs = epochs
        self.alphas = alphas
        self.sampling = sampling
        self.k_neighbors = k_neighbors

    def _fit(self, X, y):
        """
            Runs the MOS algorithm on the specified dataset.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            y : array-like, shape (n_samples)
                The classes for the samples.

            Returns
            ------
            None
        """

        if self.loss not in ['hinge', 'log']:
            raise KeyError("Loss should be 'hinge' or 'log', %r was passed" % self.loss)

        if self.sampling:
            try:
                X, y = SMOTE(random_state=self.seed, k_neighbors=self.k_neighbors).fit_resample(X, y)
            except ValueError as e:
                print('Couldn\'t perform SMOTE because n_neighbors is bigger than amount of instances in one of the classes.')
                raise(e)

        min_rvalue = 1
        min_b = []
        model = clone(self.model)
        for a in self.alphas:  # TODO: do a little more research on the range of lambdas
            model = model.set_params(loss=self.loss, random_state=self.seed, penalty='elasticnet',
                               alpha=a, l1_ratio=self.l1_ratio, max_iter=self.epochs)
            model.fit(X, y)
            b = self.weight_func(model)
            rvalue = augmented_rvalue(X[:, [i for i in range(X.shape[1]) if np.abs(b[i]) > self.threshold]], y)
            if min_rvalue > rvalue:
                min_rvalue = rvalue
                min_b = b
        self.selected_features_ = [i for i in range(X.shape[1]) if np.abs(min_b[i]) > self.threshold]
