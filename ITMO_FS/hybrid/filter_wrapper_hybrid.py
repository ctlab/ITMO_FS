from logging import getLogger

from sklearn.base import clone

from ..utils import BaseTransformer

class FilterWrapperHybrid(BaseTransformer):
    """Perform the filter + wrapper hybrid algorithm by first running the
    filter algorithm on the full dataset, leaving the selected features and
    running the wrapper algorithm on the cut dataset.

    Parameters
    ----------
    filter_ : object
        A feature selection model that should have a fit(X, y) method and a
        selected_features_ attribute available after fitting.
    wrapper : object
        A feature selection model that should have a fit(X, y) method,
        selected_features_ and best_score_ attributes available after fitting
        and a predict(X) method.

    Notes
    -----
    This class doesn't require the first algorithm to be a filter (the only
    requirements are a fit(X, y) method and a selected_features_ attribute)
    but it is recommended to use a fast algorithm first to remove a lot of
    unnecessary features before processing the resulting dataset with a more
    time-consuming algorithm (e.g. a wrapper).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from ITMO_FS.wrappers.deterministic import BackwardSelection
    >>> from ITMO_FS.filters.univariate import UnivariateFilter
    >>> from ITMO_FS.hybrid import FilterWrapperHybrid
    >>> from sklearn.datasets import make_classification
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> filter_ = UnivariateFilter('FRatio', ("K best", 10))
    >>> wrapper = BackwardSelection(LogisticRegression(), 5, measure='f1_macro')
    >>> model = FilterWrapperHybrid(filter_, wrapper).fit(x, y)
    >>> model.selected_features_
    array([ 1,  3,  4, 10,  7], dtype=int64)
    """
    def __init__(self, filter_, wrapper):
        self.filter_ = filter_
        self.wrapper = wrapper

    def _fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The classes for the samples.

        Returns
        -------
        None
        """
        self._filter = clone(self.filter_)
        self._wrapper = clone(self.wrapper)
        getLogger(__name__).info(
            "Running FilterWrapper with filter = %s, wrapper = %s",
            self._filter, self._wrapper)

        selected_filter = self._filter.fit(X, y).selected_features_
        getLogger(__name__).info(
            "Features selected by filter: %s", selected_filter)
        self.selected_features_ = selected_filter[self._wrapper.fit(
            X[:, selected_filter], y).selected_features_]
        self.best_score_ = self._wrapper.best_score_

    def predict(self, X):
        """Predict class labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        ------
        array-like, shape (n_samples,) : class labels
        """
        return self._wrapper.predict(X)
