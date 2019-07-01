#!/usr/bin/env python

import random as rnd
import numpy as np
from wrappers.wrapper_utils import get_current_accuracy



class SequentialForwardSelection:
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

        See Also
        --------


        Examples
        --------

        """

    def __init__(self, estimator, n_features, seed=1):
        self.__estimator__ = estimator
        self.__n_features__ = n_features
        self.__features__ = []
        rnd.seed(seed)

    def fit(self, X, y, test_x, test_y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_features,n_samples)
                The training input samples.
            y : array-like, shape (n_features,n_samples)
                the target values.
            test_x : array-like, shape (n_features, n_samples)
                Testing Set
            test_y : array-like , shape(n_samples)
                Labels
            Returns
            ------
            None

            See Also
            --------
            Examples
            --------

        """
        accuracy = 0
        current_features = np.array([])

        for feature in X[0:1, :]:
            old_features = current_features
            current_features = np.append(current_features, feature)
            self.__estimator__.fit([X[i] for i in current_features], y)

            current_accuracy = get_current_accuracy(X, current_features, test_x, test_y)

            if current_accuracy > accuracy:
                self.__features__ = current_features
                accuracy = current_accuracy
                if len(self.__features__) == self.__n_features__:
                    break
            else:
                current_features = old_features

    def predict(self, X):
        self.__estimator__.predict([X[i] for i in self.__features__])
