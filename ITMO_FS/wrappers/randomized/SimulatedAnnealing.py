from logging import getLogger

import numpy as np
from sklearn.model_selection import cross_val_score

from ...utils import BaseWrapper, generate_features


class SimulatedAnnealing(BaseWrapper):
    """Simulated Annealing algorithm.

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
    iteration_number : int
        Number of iterations of the algorithm.
    c : int
        A constant that is used to control the rate of feature perturbation.
    init_number_of_features : int
        The number of features to initialize start features subset with, by
        default 5-10 percents of features is used.
    cv : int
        Number of folds in cross-validation.
        
    Notes
    -----
    For more details see `this paper <http://www.feat.engineering/simulated-annealing.html/>`_.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from ITMO_FS.wrappers.randomized import SimulatedAnnealing
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> sa = SimulatedAnnealing(LogisticRegression(), measure='f1_macro',
    ... iteration_number=50).fit(x, y)
    >>> sa.selected_features_
    array([ 1,  4,  3, 17, 10, 16, 11, 14,  5], dtype=int64)
    """
    def __init__(self, estimator, measure, seed=42, iteration_number=100, c=1,
                 init_number_of_features=None, cv=3):
        self.estimator = estimator
        self.measure = measure
        self.seed = seed
        self.iteration_number = iteration_number
        self.c = c
        self.init_number_of_features = init_number_of_features
        self.cv = cv

    def __acceptance(self, i, prev_score, cur_score):
        return np.exp((i + 1) / self.c * (cur_score - prev_score) / prev_score)

    def _fit(self, X, y):
        """Fit the wrapper.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            the target values.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(self.seed)
        features = generate_features(X)

        if self.init_number_of_features is None:
            percentage = rng.integers(5, 11)
            init_number_of_features = int(
                self.n_features_ * percentage / 100) + 1
        elif self.init_number_of_features == 0:
            getLogger(__name__).warning(
                "Initial number of features was set to zero; would use one "
                "instead")
            init_number_of_features = 1
        else:
            init_number_of_features = self.init_number_of_features

        feature_subset = np.unique(
            rng.integers(0, self.n_features_, init_number_of_features))
        getLogger(__name__).info("Initial selected set: %s", feature_subset)
        prev_score = cross_val_score(
            self._estimator, X[:, feature_subset], y, cv=self.cv,
            scoring=self.measure).mean()
        getLogger(__name__).info("Initial score: %d", prev_score)

        for i in range(self.iteration_number):
            getLogger(__name__).info("Current best score: %d", prev_score)
            operation = rng.integers(0, 2)
            percentage = rng.integers(1, 5)
            if operation == 1 & feature_subset.shape[0] != self.n_features_:
                # inc
                not_included_features = np.setdiff1d(features, feature_subset)
                include_number = min(
                    not_included_features.shape[0],
                    int(self.n_features_ * (percentage / 100)) + 1)
                to_add = rng.choice(
                    not_included_features, size=include_number, replace=False)
                getLogger(__name__).info(
                    "Trying to add features %s into the selected set", to_add)
                cur_subset = np.append(feature_subset, to_add)
            else:
                # exc
                exclude_number = min(
                    feature_subset.shape[0] - 1,
                    int(self.n_features_ * (percentage / 100)) + 1)
                to_delete = rng.choice(
                    np.arange(feature_subset.shape[0]), size=exclude_number,
                    replace=False)
                getLogger(__name__).info(
                    "Trying to delete features %s from the selected set",
                    feature_subset[to_delete])
                cur_subset = np.delete(feature_subset, to_delete)
            cur_score = cross_val_score(
                self._estimator, X[:, cur_subset], y, cv=self.cv,
                scoring=self.measure).mean()
            getLogger(__name__).info("New score: %d", cur_score)
            if cur_score > prev_score:
                feature_subset = cur_subset
                prev_score = cur_score
            else:
                getLogger(__name__).info(
                    "Score has not improved; trying to accept the new subset "
                    "anyway")
                ruv = rng.random()
                acceptance = self.__acceptance(i, prev_score, cur_score)
                getLogger(__name__).info(
                    "Random value = %d, acceptance = %d", ruv, acceptance)
                if ruv < acceptance:
                    getLogger(__name__).info("Accepting the new subset")
                    feature_subset = cur_subset
                    prev_score = cur_score

        self.selected_features_ = feature_subset
        self.best_score_ = prev_score
        self._estimator.fit(X[:, self.selected_features_], y)
