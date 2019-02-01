import random as rnd

import numpy as np

from filters import RandomFilter


class RandomWrapper:
    """
        Creates random feature wrapper

        Parameters
        ----------
        estimator: object
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
        n_features : int
            Number of features to select.
        seed: int
            Seed for python random.

        See Also
        --------


        Examples
        --------

        """

    def __init__(self, estimator, n_features, seed=1):
        self.estimator_ = estimator
        self.n_features_ = n_features
        self.filter_ = RandomFilter(n_features, seed)
        self.features_ = []
        rnd.seed(seed)

    def fit(self, X, y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_features,n_samples)
                The training input samples.
            y : array-like, shape (n_features,n_samples)
                the target values.
            Returns
            ------
            None
            See Also
            --------
            Examples
            --------

        """
        self.features_ = self.filter_.run(X)
        self.estimator_.fit([X[i] for i in self.features_], y)

    def predict(self, X):
        self.estimator_.predict([X[i] for i in self.features_])


