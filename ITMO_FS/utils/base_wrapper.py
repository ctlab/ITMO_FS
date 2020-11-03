from sklearn.utils import check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from . import BaseTransformer

class BaseWrapper(BaseTransformer):
    def __init__(self):
        pass

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

        if type(X) is pd.DataFrame:
            return self._estimator.predict(X[X.columns[self.selected_features_]])
        else:
            return self._estimator.predict(X_[:, self.selected_features_])
