from sklearn.utils import check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from sklearn.base import clone

from . import BaseTransformer

class BaseWrapper(BaseTransformer):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """
            Fits the algorithm.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples), optional
                The class labels.
            fit_params : dict, optional
                Additional parameters to pass to underlying _fit function.

            Returns
            -------
            Self, i.e. the transformer object.
        """

        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self.estimator)
        if not hasattr(self.estimator, 'predict'):
            raise TypeError("estimator should be an estimator implementing "
                            "'predict' method, %r was passed" % self.estimator)
        self._estimator = clone(self.estimator)

        return super().fit(X, y, **fit_params)

    def predict(self, X):
        """
            Predicts class labels for the input data.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.

            Returns
            ------
            array-like, shape (n_samples) : class labels
        """

        check_is_fitted(self, 'selected_features_')
        X_ = check_array(X, dtype='float64', accept_large_sparse=False)
        if X_.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        return self._estimator.predict(X_[:, self.selected_features_])
