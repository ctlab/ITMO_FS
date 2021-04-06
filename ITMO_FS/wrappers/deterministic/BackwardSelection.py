#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import make_scorer

from ...utils import generate_features, BaseWrapper


class BackwardSelection(BaseWrapper):
    """
        Backward Selection removes one feature at a time until the number of features to be removed is reached. On each step,
        the best n-1 features out of n are chosen (according to some estimator metric) and the last one is removed.

        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method.
        n_features : int
            Number of features to select.
        measure : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable object / function with signature 
            measure(estimator, X, y) which should return only a single value.
        cv : int
            Number of folds in cross-validation.

        See Also
        --------


        Examples
        --------
        >>> from ITMO_FS.wrappers import BackwardSelection
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20, \
n_informative=4, n_redundant=0, shuffle=False)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> model = BackwardSelection(LogisticRegression(), 15, 'f1_macro')
        >>> model.fit()
        >>> print(model.selected_features_)
    """

    def __init__(self, estimator, n_features, measure, cv=3):
        # self.__class__ = type(estimator.__class__.__name__, (self.__class__, estimator.__class__), vars(estimator))
        self.estimator = estimator
        self.n_features = n_features
        self.measure = measure
        self.cv = cv

    def _fit(self, X, y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The training input samples.
            y : array-like, shape (n_samples,)
                The target values.
            Returns
            ------
            None
        """

        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self.estimator)

        if self.n_features > self.n_features_:
            raise ValueError("Cannot select %d features with n_features = %d" % (self.n_features, self.n_features_))

        self._estimator = clone(self.estimator)

        self.selected_features_ = generate_features(X)

        while len(self.selected_features_) != self.n_features:
            max_measure = 0
            to_delete = 0
            for i in range(len(self.selected_features_)):
                iteration_features = np.delete(self.selected_features_, i)
                iteration_measure = cross_val_score(self._estimator, X[:, iteration_features], y, cv=self.cv,
                                                    scoring=self.measure).mean()
                if iteration_measure > max_measure:
                    max_measure = iteration_measure
                    to_delete = i
            self.selected_features_ = np.delete(self.selected_features_, to_delete)
        self.best_score_ = cross_val_score(self._estimator, X[:, self.selected_features_], y, cv=self.cv,
                                                    scoring=self.measure).mean()
        self._estimator.fit()
