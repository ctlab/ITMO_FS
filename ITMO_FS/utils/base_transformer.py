from sklearn.utils import check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from abc import abstractmethod


class BaseTransformer(TransformerMixin, BaseEstimator):
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

        if y is not None:
            X, y = check_X_y(X, y, dtype='numeric')
            if y.dtype.kind == 'O':
                y = y.astype('int')
        else:
            X = check_array(X, dtype='float64', accept_large_sparse=False)

        self.n_total_features_ = X.shape[1]
        nonconst_features = VarianceThreshold().fit(X).get_support(indices=True)
        self.n_features_ = nonconst_features.shape[0]
        #TODO: log warning if nonconst_features.shape[0] != self.n_total_features
        self._fit(X[:, nonconst_features], y, **fit_params)

        if hasattr(self, 'feature_scores_'):
            scores = np.empty(self.n_features_)
            scores.fill(np.nan)
            scores[nonconst_features] = self.feature_scores_
        self.selected_features_ = nonconst_features[self.selected_features_]

        return self

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            ------
            Transformed 2D numpy array
        """
        check_is_fitted(self, 'selected_features_')
        X_ = check_array(X, dtype='numeric', accept_large_sparse=False)
        if X_.shape[1] != self.n_total_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        if isinstance(X, pd.DataFrame):
            return X[X.columns[self.selected_features_]]
        else:
            return X_[:, self.selected_features_]

    @abstractmethod
    def _fit(self, X, y):
        pass
