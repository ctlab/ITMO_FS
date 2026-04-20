from abc import abstractmethod
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class BaseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit the algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,), optional
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

        if self.n_features_ != self.n_total_features_:
            getLogger(__name__).warning(
                "Found %d constant features; they would not be used in fit")

        if hasattr(self, 'n_features'):
            if self.n_features > self.n_features_:
                getLogger(__name__).error(
                    "Cannot select %d features with n_features = %d",
                    self.n_features, self.n_features_)
                raise ValueError(
                    "Cannot select %d features with n_features = %d"
                    % (self.n_features, self.n_features_))

        if hasattr(self, 'epsilon'):
            if self.epsilon <= 0:
                getLogger(__name__).error(
                    "Epsilon should be positive, %d passed", self.epsilon)
                raise ValueError(
                    "Epsilon should be positive, %d passed" % self.epsilon)


        self._fit(X[:, nonconst_features], y, **fit_params)

        if hasattr(self, 'feature_scores_'):
            scores = np.empty(self.n_total_features_)
            scores.fill(np.nan)
            scores[nonconst_features] = self.feature_scores_
            self.feature_scores_ = scores
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
            getLogger(__name__).error(
                "Shape of input is different from what was seen in 'fit'")
            raise ValueError(
                "Shape of input is different from what was seen in 'fit'")
        if isinstance(X, pd.DataFrame):
            return X[X.columns[self.selected_features_]]
        else:
            return X_[:, self.selected_features_]

    @abstractmethod
    def _fit(self, X, y):
        pass
