from logging import getLogger

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ...utils import generate_features, BaseWrapper

class HillClimbingWrapper(BaseWrapper):
    """Hill Climbing algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method.
    measure : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value.
    seed : int
        Random seed used to initialize np.random.default_rng().
    cv : int
        Number of folds in cross-validation.

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.wrappers import HillClimbingWrapper
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> model = HillClimbingWrapper(LogisticRegression(),
    ... measure='f1_macro').fit(x, y)
    >>> model.selected_features_
    array([ 0,  1,  2,  3,  4,  6,  7,  9, 11, 13, 14, 15], dtype=int64)
    """
    def __init__(self, estimator, measure, seed=42, cv=3):
        self.estimator = estimator
        self.measure = measure
        self.seed = seed
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
        rng = np.random.default_rng(self.seed)

        features = generate_features(X)
        mask = rng.choice([True, False], self.n_features_)
        getLogger(__name__).info("Initial feature mask: %s", mask)
        score = cross_val_score(
            self._estimator, X[:, features[mask]], y, cv=self.cv,
            scoring=self.measure).mean()

        while True:
            getLogger(__name__).info("Current best score: %d", score)
            old_score = score
            order = rng.permutation(self.n_features_)
            for feature in order:
                getLogger(__name__).info("Trying to change feature %d", feature)
                mask[feature] = not(mask[feature])
                new_score = cross_val_score(self._estimator,
                    X[:, features[mask]], y, cv=self.cv,
                    scoring=self.measure).mean()
                getLogger(__name__).info("New score: %d", new_score)
                if new_score > score:
                    score = new_score
                    break
                mask[feature] = not(mask[feature])
            if old_score == score:
                break

        self.selected_features_ = features[mask]
        self.best_score_ = score
        self._estimator.fit(X[:, self.selected_features_], y)
