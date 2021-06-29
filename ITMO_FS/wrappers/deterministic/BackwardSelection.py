from logging import getLogger

import numpy as np
from sklearn.model_selection import cross_val_score

from ...utils import generate_features, BaseWrapper


class BackwardSelection(BaseWrapper):
    """Backward Selection removes one feature at a time until the number of
    features to be removed is reached. On each step, the best n-1 features
    out of n are chosen (according to some estimator metric) and the last one
    is removed.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method.
    n_features : int
        Number of features to select.
    measure : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value.
    cv : int
        Number of folds in cross-validation.

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.wrappers import BackwardSelection
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> model = BackwardSelection(LogisticRegression(), 5,
    ... measure='f1_macro').fit(x, y)
    >>> model.selected_features_
    array([ 0,  1,  2,  3, 13], dtype=int64)
    """
    def __init__(self, estimator, n_features, measure, cv=3):
        self.estimator = estimator
        self.n_features = n_features
        self.measure = measure
        self.cv = cv

    def _fit(self, X, y):
        """Fit the wrapper.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        self.selected_features_ = generate_features(X)

        while self.selected_features_.shape[0] != self.n_features:
            getLogger(__name__).info(
                "Current selected set: %s", self.selected_features_)
            scores = np.vectorize(
                lambda i: cross_val_score(
                    self._estimator,
                    X[:, np.delete(self.selected_features_, i)], y, cv=self.cv,
                    scoring=self.measure).mean())(
            np.arange(0, self.selected_features_.shape[0]))
            getLogger(__name__).info(
                "Scores for all selected features: %s", scores)
            to_delete = np.argmax(scores)
            getLogger(__name__).info(
                "Deleting feature %d", self.selected_features_[to_delete])

            self.selected_features_ = np.delete(
                self.selected_features_, to_delete)
        self.best_score_ = cross_val_score(
            self._estimator, X[:, self.selected_features_], y, cv=self.cv,
            scoring=self.measure).mean()
        self._estimator.fit(X[:, self.selected_features_], y)
