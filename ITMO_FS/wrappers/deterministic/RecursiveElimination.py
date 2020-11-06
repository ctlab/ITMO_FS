#!/usr/bin/env python

import numpy as np

from ...utils import generate_features


class RecursiveElimination:
    """
        Performs a recursive feature elimination until the required number of features is reached.

        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
        n_features : int
            Number of features to leave.

        See Also
        --------

        Examples
        --------

        """

    def __init__(self, estimator, n_features):
        if not hasattr(estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % estimator)

        self.__estimator__ = estimator
        self.__n_features__ = n_features
        self.__features__ = []

    def fit(self, X, y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The training input samples.
            y : array-like, shape (n_samples,)
                the target values.
            Returns
            ------
            None

            See Also
            --------
            Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., “Gene selection for cancer classification using support vector machines”, Mach. Learn., 46(1-3), 389–422, 2002.
            https://link.springer.com/article/10.1023/A:1012487302797

            Examples
            --------
            >>> from sklearn.datasets import make_classification
            >>> from ITMO_FS.wrappers import RecursiveElimination
            >>> from sklearn.svm import SVC
            >>> import numpy as np
            >>> dataset = make_classification(n_samples=1000, n_features=20)
            >>> data, target = np.array(dataset[0]), np.array(dataset[1])
            >>> model = SVC(kernel='linear')
            >>> rfe = RecursiveElimination(model, 5)
            >>> rfe.fit(data, target)
            >>> print("Resulting features: ", rfe.__features__)

        """
        self.__features__ = generate_features(X)

        while len(self.__features__) != self.__n_features__:
            self.__estimator__.fit(X[:, self.__features__], y)

            if hasattr(self.__estimator__, 'coef_'):
                coefs = np.square(self.__estimator__.coef_)
            elif hasattr(self.__estimator__, 'feature_importances_'):
                coefs = np.square(self.__estimator__.feature_importances_)
            else:
                raise TypeError("estimator should be an estimator with a "
                                "'coef_' or 'feature_importances_' attribute, %r was passed" % self.__estimator__)
            if coefs.ndim > 1:
                coefs = coefs.sum(axis=0)

            least_important = np.argmin(coefs)

            self.__features__ = np.delete(self.__features__, least_important)

    def predict(self, X):
        self.__estimator__.predict(X[:, self.__features__])
