from logging import getLogger
import random as rnd

import numpy as np
from sklearn.model_selection import cross_val_score

from ...utils import BaseWrapper, generate_features


class AddDelWrapper(BaseWrapper):
    """Add-Del feature wrapper.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method.
    measure : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value.
    cv : int
        Number of folds in cross-validation.
    seed : int
        Seed for python random.
    d : int
        Amount of consecutive iterations for add ana del procedures that can
        have decreasing objective function before the algorithm terminates.

    See Also
    --------
    Lecture about feature selection (ru), p.13 -
    http://www.ccas.ru/voron/download/Modeling.pdf

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> lg = LogisticRegression(solver='lbfgs')
    >>> add_del = AddDelWrapper(lg, 'accuracy').fit(x, y)
    >>> add_del.selected_features_
    array([1, 4, 3], dtype=int64)
    """
    def __init__(self, estimator, measure, cv=3, seed=42, d=1):
        self.estimator = estimator
        self.measure = measure
        self.cv = cv
        self.seed = seed
        self.d = d

    def __add(self, X, y, free_features):
        """Add features to the selected set one by one until either all of
        the features are added or more than d iterations pass without
        increasing the objective function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        free_features : array-like, shape (n_not_selected_features,)
            The array of current free features.

        Returns
        -------
        array-like, shape (n_new_selected_features,) : selected features;
        array-like, shape (n_new_not_selected_features,) : new free features
        """
        best_score = self.best_score_
        iteration_features = self.selected_features_
        iteration_free_features = free_features
        selected_features = self.selected_features_
        getLogger(__name__).info(
            "Trying to add features from free set %s to selected set %s",
            free_features, selected_features)

        while (iteration_features.shape[0] - selected_features.shape[0] <=
                   self.d) & (iteration_free_features.shape[0] != 0):
            getLogger(__name__).info(
                "Current selected set: %s, best score: %d",
                selected_features, best_score)
            scores = np.vectorize(
                lambda f: cross_val_score(
                    self._estimator, X[:, np.append(iteration_features, f)], y,
                    cv=self.cv, scoring=self.measure).mean())(
            iteration_free_features)
            getLogger(__name__).info("Scores for all free features: %s", scores)

            to_add = np.argmax(scores)
            iteration_score = scores[to_add]
            getLogger(__name__).info(
                "Adding feature %d, new score: %d",
                iteration_free_features[to_add], iteration_score)
            iteration_features = np.append(
                iteration_features, iteration_free_features[to_add])
            iteration_free_features = np.delete(iteration_free_features, to_add)

            if iteration_score > best_score:
                selected_features = iteration_features
                free_features = iteration_free_features
                best_score = iteration_score

        return selected_features, free_features

    def __del(self, X, y, selected_features, free_features):
        """Delete features from the selected set one by one until either only
        one feature is left or more than d iterations pass without
        increasing the objective function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        selected_features : array-like, shape (n_selected_features,)
            The array of current selected features.
        free_features : array-like, shape (n_not_selected_features,)
            The array of current free features.

        Returns
        -------
        array-like, shape (n_new_selected_features,) : new selected features;
        array-like, shape (n_new_not_selected_features,) : new free features;
        float : score for the selected feature set
        """
        best_score = cross_val_score(
            self._estimator, X[:, selected_features], y, scoring=self.measure,
            cv=self.cv).mean()
        iteration_features = selected_features
        iteration_free_features = free_features
        getLogger(__name__).info(
            "Trying to delete features from selected set %s", selected_features)

        while (selected_features.shape[0] - iteration_features.shape[0] <=
                   self.d) & (iteration_features.shape[0] != 1):
            getLogger(__name__).info(
                "Current selected set: %s, best score: %d",
                selected_features, best_score)
            scores = np.vectorize(
                lambda i: cross_val_score(
                    self._estimator, X[:, np.delete(iteration_features, i)], y,
                    cv=self.cv, scoring=self.measure).mean())(
            np.arange(0, iteration_features.shape[0]))
            getLogger(__name__).info(
                "Scores for all selected features: %s", scores)

            to_delete = np.argmax(scores)
            iteration_score = scores[to_delete]
            getLogger(__name__).info(
                "Deleting feature %d, new score: %d",
                iteration_features[to_delete], iteration_score)
            iteration_free_features = np.append(
                iteration_free_features, iteration_features[to_delete])
            iteration_features = np.delete(iteration_features, to_delete)

            if iteration_score > best_score:
                selected_features = iteration_features
                free_features = iteration_free_features
                best_score = iteration_score

        return selected_features, free_features, best_score

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
        self.selected_features_ = np.array([], dtype='int')
        free_features = generate_features(X)
        self.best_score_ = 0
        while True:
            selected_features, free_features = self.__add(X, y, free_features)
            getLogger(__name__).info(
                "After add: selected set = %s, free set = %s",
                selected_features, free_features)
            selected_features, free_features, iteration_score = self.__del(
                X, y, selected_features, free_features)
            getLogger(__name__).info(
                "After del: selected set = %s, free set = %s, score = %d",
                selected_features, free_features, iteration_score)

            if iteration_score > self.best_score_:
                self.best_score_ = iteration_score
                self.selected_features_ = selected_features
            else:
                break
        self._estimator.fit(X[:, self.selected_features_], y)
