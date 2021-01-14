#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from ...utils import generate_features, BaseWrapper

class SequentialForwardSelection(BaseWrapper):
    """
        Sequentially Adds Features that Maximises the Classifying function when combined with the features already used
        TODO add theory about this method

        Parameters
        ----------
        estimator: object
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
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
        >>> from ITMO_FS.wrappers import SequentialForwardSelection
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> model = SequentialForwardSelection(LogisticRegression(), 5, 'f1_macro')
        >>> model.fit(data, target)
        >>> print(model.selected_features_)
    """

    def __init__(self, estimator, n_features, measure, cv=3):
        self.estimator = estimator
        self.n_features = n_features
        self.measure = measure
        self.cv = cv

    def _fit(self, X, y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_features,n_samples)
                The training input samples.
            y : array-like, shape (n_features,n_samples)
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

        self.selected_features_ = np.array([], dtype=int)
        features_left = generate_features(X)

        while len(self.selected_features_) != self.n_features:
            max_measure = 0
            to_add = 0
            for i in range(len(features_left)):
                iteration_features = np.append(self.selected_features_, features_left[i])
                iteration_measure = cross_val_score(self._estimator, X[:, iteration_features], y, cv=self.cv,
                                                    scoring=self.measure).mean()
                if iteration_measure > max_measure:
                    max_measure = iteration_measure
                    to_add = i
            self.selected_features_ = np.append(self.selected_features_, features_left[to_add])
            features_left = np.delete(features_left, to_add)
        self.best_score_ = cross_val_score(self._estimator, X[:, self.selected_features_], y, cv=self.cv,
                                                    scoring=self.measure).mean()
        self._estimator.fit(X[:, self.selected_features_], y)
