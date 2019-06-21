#!/usr/bin/env python

from collections import OrderedDict
import numpy as np

from wrappers.wrapper_utils import get_current_accuracy


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


        Examples
        --------

        """

    def __init__(self, estimator, n_features):
        self.__estimator__ = estimator
        self.__n_features__ = n_features
        self.__features__ = []


    def fit(self, X, y, test_x, test_y, feature_ranks):
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
            feature_ranks : dict,
                Contains features index and its ranks
            Returns
            ------
            None

            See Also
            --------
            Examples
            --------

        """
        sorted_features_ranks = OrderedDict(sorted(feature_ranks.items(), key=lambda x: x[1]))
        selected_features = np.array([feature for feature in sorted_features_ranks])
        number_of_features_left_to_remove = self.__n_features__

        self.__estimator__.fit([X[i] for i in selected_features], y)
        accuracy = get_current_accuracy((self.__estimator__, X, selected_features, test_x, test_y))
        i = 0
        while len(sorted_features_ranks) != i:
            iteration_features = np.delete(selected_features, (i))
            self.__estimator__.fit([X[i] for i in iteration_features], y)

            iteration_accuracy = get_current_accuracy(self.__estimator__, X, iteration_features, test_x, test_y)
            if iteration_accuracy > accuracy:
                selected_features = iteration_features
                number_of_features_left_to_remove -= 1
                if not number_of_features_left_to_remove:
                    break
            else:
                i +=1

        self.__features__ = selected_features



    def predict(self, X):
        self.__estimator__.predict([X[i] for i in self.__features__])
