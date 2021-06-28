from logging import getLogger

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

from ..utils import augmented_rvalue, BaseTransformer


class MOS(BaseTransformer):
    """Perform Minimizing Overlapping Selection under SMOTE (MOSS) or under
    No-Sampling (MOSNS) algorithm.

    Parameters
    ----------
    model : object
        The model that should have a fit(X, y) method and a field corresponding
        to feature weights. Currently only SGDClassifier should be passed,
        other models would not work.
    weight_func : callable
        The function to extract weights from the model.
    loss : str, 'log' or 'hinge'
        Loss function to use in the algorithm. 'log' gives a logistic
        regression, while 'hinge' gives a support vector machine.
    seed : int, optional
        Seed for python random.
    l1_ratio : float
        The value used to balance the L1 and L2 penalties in elastic-net.
    threshold : float
        The threshold value for feature dropout. Instead of comparing them to
        zero, they are normalized and values with absolute value lower than the
        threshold are dropped out.
    epochs : int
        The number of epochs to perform in the algorithm.
    alphas : array-like, shape (n_alphas,), optional
        The range of lambdas that should form the regularization path.
    sampling : bool
        Bool value that control whether MOSS (True) or MOSNS (False) should be
        executed.
    k_neighbors : int
        Amount of nearest neighbors to use in SMOTE if MOSS is used.

    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/pii/S0169743919306070/>`_.

    Examples
    --------
    >>> from ITMO_FS.embedded import MOS
    >>> from sklearn.linear_model import SGDClassifier
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> dataset = make_classification(n_samples=100, n_features=10,
    ... n_informative=5, n_redundant=0, weights=[0.85, 0.15], random_state=42,
    ... shuffle=False)
    >>> X, y = np.array(dataset[0]), np.array(dataset[1])
    >>> m = MOS(model=SGDClassifier(),
    ... weight_func=lambda model: np.square(model.coef_).sum(axis=0)).fit(X, y)
    >>> m.selected_features_
    array([1, 3, 4], dtype=int64)
    >>> m = MOS(model=SGDClassifier(), sampling=True,
    ... weight_func=lambda model: np.square(model.coef_).sum(axis=0)).fit(X, y)
    >>> m.selected_features_
    array([1, 3, 4, 6], dtype=int64)
    """
    def __init__(self, model, weight_func, loss='log', seed=42, l1_ratio=0.5,
                 threshold=1e-3, epochs=1000, alphas=np.arange(0.01, 0.2, 0.01),
                 sampling=False, k_neighbors=2):
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
        """Run the MOS algorithm on the specified dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The classes for the samples.

        Returns
        -------
        None
        """
        if self.loss not in ['hinge', 'log']:
            getLogger(__name__).error(
                "Loss should be 'hinge' or 'log', %s was passed", self.loss)
            raise KeyError(
                "Loss should be 'hinge' or 'log', %s was passed" % self.loss)

        if self.sampling:
            try:
                X, y = SMOTE(
                    random_state=self.seed,
                    k_neighbors=self.k_neighbors).fit_resample(X, y)
            except ValueError:
                getLogger(__name__).warning(
                    "Couldn't perform SMOTE because k_neighbors is bigger "
                    "than amount of instances in one of the classes; MOSNS "
                    "would be performed instead")

        min_rvalue = 1
        min_b = []
        model = clone(self.model)
        for a in self.alphas:  # TODO: do a little more
            # research on the range of lambdas
            model = model.set_params(
                loss=self.loss, random_state=self.seed, penalty='elasticnet',
                alpha=a, l1_ratio=self.l1_ratio, max_iter=self.epochs)
            model.fit(X, y)
            b = self.weight_func(model)
            rvalue = augmented_rvalue(
                X[:, np.flatnonzero(np.abs(b) > self.threshold)], y)
            getLogger(__name__).info(
                "For alpha %f: rvalue = %f, weight vector = %s", a, rvalue, b)
            if min_rvalue > rvalue:
                min_rvalue = rvalue
                min_b = b
                getLogger(__name__).info("New minimum rvalue: %f", rvalue)
                getLogger(__name__).info("New weight vector: %s", b)
        self.selected_features_ = np.flatnonzero(np.abs(min_b) > self.threshold)
