#!/usr/bin/env python

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ...utils import generate_features, BaseWrapper


class RecursiveElimination(BaseWrapper):
    """
        Performs a recursive feature elimination until the required number of features is reached.

        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
        n_features : int
            Number of features to leave.
        measure : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable object / function with signature
            measure(estimator, X, y) which should return only a single value.
        cv : int
            Number of folds in cross-validation.
            
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
        >>> rfe.fit()
        >>> print("Resulting features: ", rfe.selected_features_)
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
            X : array-like, shape (n_samples,n_features)
                The training input samples.
            y : array-like, shape (n_samples,)
                the target values.
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
            self._estimator.fit()

            if hasattr(self._estimator, 'coef_'):
                coefs = np.square(self._estimator.coef_)
            elif hasattr(self._estimator, 'feature_importances_'):
                coefs = np.square(self._estimator.feature_importances_)
            else:
                raise TypeError("estimator should be an estimator with a "
                                "'coef_' or 'feature_importances_' attribute, %r was passed" % self._estimator)
            if coefs.ndim > 1:
                coefs = coefs.sum(axis=0)

            least_important = np.argmin(coefs)

            self.selected_features_ = np.delete(self.selected_features_, least_important)
        self.best_score_ = cross_val_score(self._estimator, X[:, self.selected_features_], y, cv=self.cv,
                                                scoring=self.measure).mean()
        self._estimator.fit()
