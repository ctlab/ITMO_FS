from sklearn.utils import check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd

class DataChecker:
    def __init__(self):
        pass

class BaseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):

        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)

        self.n_features_ = X.shape[1]
        self._fit(X, y, **fit_params)

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
        X_ = check_array(X)
        if X_.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        if type(X) is pd.DataFrame:
            return X[X.columns[self.selected_features_]]
        else:
            return X_[:, self.selected_features_]
