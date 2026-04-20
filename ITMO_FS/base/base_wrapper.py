from logging import getLogger

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from . import BaseTransformer

class BaseWrapper(BaseTransformer):
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
        if not hasattr(self.estimator, 'fit'):
            getLogger(__name__).error(
                "estimator should be an estimator implementing "
                "'fit' method, %s was passed", self.estimator)
            raise TypeError(
                "estimator should be an estimator implementing "
                "'fit' method, %s was passed" % self.estimator)
        if not hasattr(self.estimator, 'predict'):
            getLogger(__name__).error(
                "estimator should be an estimator implementing "
                "'predict' method, %s was passed", self.estimator)
            raise TypeError(
                "estimator should be an estimator implementing "
                "'predict' method, %s was passed" % self.estimator)
        self._estimator = clone(self.estimator)

        return super().fit(X, y, **fit_params)

    def predict(self, X):
        """Predict class labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array-like, shape (n_samples,) : class labels
        """
        check_is_fitted(self, 'selected_features_')
        X_ = check_array(X, dtype='float64', accept_large_sparse=False)
        if X_.shape[1] != self.n_features_:
            getLogger(__name__).error(
                "Shape of input is different from what was seen in 'fit'")
            raise ValueError(
                "Shape of input is different from what was seen in 'fit'")

        return self._estimator.predict(X_[:, self.selected_features_])
