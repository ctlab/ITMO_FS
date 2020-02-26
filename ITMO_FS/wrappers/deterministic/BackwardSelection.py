#!/usr/bin/env python

from collections import OrderedDict

import numpy as np

from ..utils import generate_features
from .wrapper_utils import get_current_cv_accuracy


class BackwardSelection:
    """
        Backward Selection removes one feature at a time until the number of features to be removed are used. Which ever
        feature has the least rank it is removed one by one.
        Parameters
        ----------
        estimator: object
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
        n_features : int
            Number of features to be removed.

        See Also
        --------


        examples
        --------

        """

    def __init__(self, estimator, n_features, measure):
        # self.__class__ = type(estimator.__class__.__name__, (self.__class__, estimator.__class__), vars(estimator))
        if not hasattr(estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % estimator)
        self._estimator = estimator
        self.__n_features__ = n_features
        self.features__ = []
        self.__measure = measure
        self.best_score = None

    def fit(self, X, y, cv=3):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The training input samples.
            y : array-like, shape (n_samples,)
                the target values.
            cv : int
                Number of folds in cross-validation
            Returns
            ------
            None

            See Also
            --------
            examples
            --------

        """
        features_ranks = dict(zip(generate_features(X), self.__measure(X, y)))
        sorted_features_ranks = OrderedDict(sorted(features_ranks.items(), key=lambda x: x[1]))
        selected_features = np.array([feature for feature in sorted_features_ranks])
        number_of_features_left_to_remove = self.__n_features__

        self._estimator.fit(X[:, selected_features], y)
        accuracy = get_current_cv_accuracy(self._estimator, X, y, selected_features, cv)
        i = 0
        self.best_score = accuracy
        while len(sorted_features_ranks) != i and i < len(selected_features):
            iteration_features = np.delete(selected_features, i)
            self._estimator.fit(X[:, iteration_features], y)

            iteration_accuracy = get_current_cv_accuracy(self._estimator, X, y, iteration_features, cv)
            if iteration_accuracy > self.best_score:
                selected_features = iteration_features
                number_of_features_left_to_remove -= 1
                self.best_score = iteration_accuracy
                if not number_of_features_left_to_remove:
                    break
            else:
                i += 1

        self.features__ = selected_features

    def predict(self, X):
        self._estimator.predict(X[:, self.features__])
