from logging import getLogger

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.utils import BaseWrapper, apply_cr
from ITMO_FS.utils.data_check import *


class Melif(BaseWrapper):
    """MeLiF algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method.
    measure : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value.
    cutting_rule : string or callable
        A cutting rule name defined in GLOB_CR or a callable with signature
        cutting_rule (features), which should return a list features ranked by
        some rule.
    filter_ensemble : object
        A filter ensemble (e.g. WeightBased) or a list of filters that will be
        used to create a WeightBased ensemble.
    delta : float
        The step in coordinate descent.
    points : array-like
        An array of starting points in the search.
    seed : int
        Random seed used to initialize np.random.default_rng().
    cv : int
        Number of folds in cross-validation.

    See Also
    --------
    For more details see `this paper <https://www.researchgate.net/publication/317201206_MeLiF_Filter_Ensemble_Learning_Algorithm_for_Gene_Selection>`_.

    Examples
    --------
    >>> from ITMO_FS.hybrid import Melif
    >>> from ITMO_FS.filters.univariate import UnivariateFilter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> x = KBinsDiscretizer(n_bins=10, encode='ordinal',
    ... strategy='uniform').fit_transform(x)
    >>> filters = [UnivariateFilter('GiniIndex'),
    ... UnivariateFilter('FechnerCorr'),
    ... UnivariateFilter('SpearmanCorr'),
    ... UnivariateFilter('PearsonCorr')]
    >>> algo = Melif(LogisticRegression(), 'f1_macro', ("K best", 5),
    ... filters, delta=0.5).fit(x, y)
    >>> algo.selected_features_
    array([ 3,  4,  1, 13, 16], dtype=int64)
    """
    def __init__(self, estimator, measure, cutting_rule, filter_ensemble,
                 delta=0.5, points=None, seed=42, cv=3):
        self.estimator = estimator
        self.measure = measure
        self.cutting_rule = cutting_rule
        self.filter_ensemble = filter_ensemble
        self.delta = delta
        self.points = points
        self.seed = seed
        self.cv = cv

    def _fit(self, X, y):
        """Run the MeLiF algorithm on the specified dataset.

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
        self._rng = np.random.default_rng(self.seed)
        if type(self.filter_ensemble) is list:
            self.__ensemble = WeightBased(self.filter_ensemble)
        else:
            self.__ensemble = clone(self.filter_ensemble)

        self.n_filters = len(self.__ensemble)
        self.__filter_weights = np.ones(self.n_filters) / self.n_filters

        check_cutting_rule(self.cutting_rule)
        cutting_rule = apply_cr(self.cutting_rule)
        getLogger(__name__).info(
            "Using MeLiF with ensemble: %s and cutting rule: %s",
            self.__ensemble, cutting_rule)
        scores = self.__ensemble.get_scores(X, y)

        if self.points is None:
            points = np.vstack((self.__filter_weights, np.eye(self.n_filters)))
        else:
            points = self.points
        best_point_ = points[0]

        self.best_score_ = 0
        for point in points:
            getLogger(__name__).info(
                "Running coordinate descent from point %s", point)
            new_point, new_score = self.__search(
                X, y, point, scores, cutting_rule)
            getLogger(__name__).info(
                "Ended up in point %s with score %d", new_point, new_score)
            if new_score > self.best_score_:
                self.best_score_ = new_score
                self.best_point_ = new_point
        getLogger(__name__).info(
            "Final best point: %s with score %d",
            best_point_, self.best_score_)
        self.selected_features_ = cutting_rule(
            np.dot(scores.T, self.best_point_))
        self._estimator.fit(X[:, self.selected_features_], y)

    def __search(self, X, y, point, scores, cutting_rule):
        """Perform a coordinate descent from the given point.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The classes for the samples.
        point : array-like, shape (n_filters,)
            The starting point.
        scores : array-like, shape (n_filters, n_features)
            The scores for the features from all filters.
        cutting_rule : callable
            The cutting rule to use.

        Returns
        -------
        tuple (array-like, float) : the optimal point and its score
        """
        best_point = point
        selected_features = cutting_rule(np.dot(scores.T, point))
        best_score = cross_val_score(
            self._estimator, X[:, selected_features], y, cv=self.cv,
            scoring=self.measure).mean()
        delta = np.eye(self.n_filters) * self.delta
        changed = True
        while changed:
            #the original paper descends starting from the first filter;
            #we randomize the order instead to avoid local maximas
            getLogger(__name__).info(
                "Current optimal point: %s with score = %d",
                best_point, best_score)
            order = self._rng.permutation(self.n_filters)
            changed = False
            for f in order:
                iteration_point_plus = best_point + delta[f]
                selected_features = cutting_rule(
                    np.dot(scores.T, iteration_point_plus))
                score = cross_val_score(
                    self._estimator, X[:, selected_features], y, cv=self.cv,
                    scoring=self.measure).mean()
                getLogger(__name__).info(
                    "Trying to move to point %s: score = %d",
                    iteration_point_plus, score)
                if score > best_score:
                    best_score = score
                    best_point = iteration_point_plus
                    changed = True
                    break

                iteration_point_minus = best_point - delta[f]
                selected_features = cutting_rule(
                    np.dot(scores.T, iteration_point_minus))
                score = cross_val_score(
                    self._estimator, X[:, selected_features], y, cv=self.cv,
                    scoring=self.measure).mean()
                getLogger(__name__).info(
                    "Trying to move to point %s: score = %d",
                    iteration_point_minus, score)
                if score > best_score:
                    best_score = score
                    best_point = iteration_point_minus
                    changed = True
                    break
        return best_point, best_score
