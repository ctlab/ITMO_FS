from logging import getLogger

import numpy as np
from sklearn.model_selection import cross_val_score

from ITMO_FS.filters.univariate.measures import su_measure, relief_measure
from ITMO_FS.utils import BaseWrapper


class IWSSr_SFLA(BaseWrapper):
    """IWSSr-SFLA (Incremental Wrapper Subset Selection with replacement and
    Shuffled Frog Leaping Algorithm).

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method.
    measure_iwssr : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value that would be used in the IWSSr algorithm.
    measure_frogs : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value that would be used to measure the frog's fitness value.
    cv : int
        Number of folds in cross-validation.
    relief_iterations : int
        Amount of iterations to do in the Relief algorithm.
    sfla_m : int
        Amount of memplexes in the SFLA algorithm.
    sfla_n : int
        Amount of frogs in each memplex in the SFLA algorithm.
    sfla_q : int
        Amount of frogs in each submemplex in the SFLA algorithm. Should be
        lower than sfla_n.
    s : int
        Maximum amount of features that can change during one leap in the SFLA
        algorithm.
    iterations : int
        Total iteration number in the SFLA algorithm.
    iterations_leaps : int
        Total amount of leaps that each memplex would try to do during each
        iteration.
    seed : int
        Python random seed.

    See Also
    --------
    For more details see `this paper <https://www.nature.com/articles/s41598-019-54987-1/>`_.

    Examples
    --------
    >>> from ITMO_FS.hybrid import IWSSr_SFLA
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> x = KBinsDiscretizer(n_bins=10, encode='ordinal',
    ... strategy='uniform').fit_transform(x)
    >>> algo = IWSSr_SFLA(LogisticRegression()).fit(x, y)
    >>> algo.selected_features_
    array([ 1,  3,  4, 10, 13, 15, 17], dtype=int64)
    """
    def __init__(self, estimator, measure_iwssr='accuracy',
                 measure_frogs='accuracy', cv=3, relief_iterations=None,
                 sfla_m=5, sfla_n=20, sfla_q=8, s=3, iterations=20,
                 iterations_leaps=10, seed=42):
        self.estimator = estimator
        self.measure_iwssr = measure_iwssr
        self.measure_frogs = measure_frogs
        self.cv = cv
        self.relief_iterations = relief_iterations
        self.sfla_m = sfla_m
        self.sfla_n = sfla_n
        self.sfla_q = sfla_q
        self.s = s
        self.iterations = iterations
        self.iterations_leaps = iterations_leaps
        self.seed = seed

    def __generate_frog(self, weights):
        """Generate a frog from the given feature weights.

        Parameters
        ----------
        weights : array-like, shape (n_features,)
            Probabilities for the features.

        Returns
        -------
        array-like, shape (n_features,) : generated frog in the form of a
        boolean array
        """
        f_num = self._rng.integers(self.n_features_) + 1
        features = self._rng.choice(
            self.n_features_, size=f_num, replace=False, p=weights)
        frog = np.full(self.n_features_, False)
        frog[features] = True
        return frog

    def __create_memplexes(self):
        """Create sfla_m memplexes.

        Parameters
        ----------

        Returns
        -------
        array-like, shape (sfla_m, sfla_n) : the indices of frogs in each of
        the memplexes
        """
        return self._rng.permutation(
            self.sfla_m * self.sfla_n).reshape((self.sfla_m, self.sfla_n))

    def __iwssr(self, X, y, frog):
        """Perform the IWSSr algorithm for given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        frog : array-like, shape (n_features,)
            A frog (boolean array) of selected features.

        Returns
        -------
        array-like, shape (n_selected_features,) : selected features
        """
        def __check_subset(subset):
            nonlocal best_measure
            nonlocal best_iteration_set

            measure = cross_val_score(
                self._estimator, X[:, subset], y, cv=self.cv,
                scoring=self.measure_iwssr).mean()
            if measure > best_measure:
                best_measure = measure
                best_iteration_set = subset

        frog_features = np.flatnonzero(frog)
        getLogger(__name__).info("Performing IWSSr for frog %s", frog_features)

        weights = su_measure(X, y)
        getLogger(__name__).info("Feature weights: %s", weights)
        order = np.argsort(weights)[::-1]
        selected_features = np.array([order[0]])
        best_measure = cross_val_score(
            self._estimator, X[:, selected_features], y, cv=self.cv,
            scoring=self.measure_iwssr).mean()
        best_iteration_set = selected_features
        getLogger(__name__).info(
            "Optimal feature set: %s, score: %d",
            best_iteration_set, best_measure)

        for feature in order[1:]:
            getLogger(__name__).info(
                "Trying to add feature %d into the set", feature)
            if selected_features.shape[0] != 1:
                for i in range(selected_features.shape[0]):
                    __check_subset(
                        np.append(
                            np.delete(selected_features, i),
                            feature))
            __check_subset(np.append(selected_features, feature))
            getLogger(__name__).info(
                "Optimal feature set: %s, score: %d",
                best_iteration_set, best_measure)
            selected_features = best_iteration_set

        selected_features = frog_features[selected_features]
        getLogger(__name__).info("New selected features: %s", selected_features)

        new_frog = np.full(self.n_features_, False)
        new_frog[selected_features] = True
        return new_frog

    def __create_submemplex(self, fitness_values):
        """Create a submemplex from the memplex of frogs.

        Parameters
        ----------
        fitness_values : array-like, shape (sfla_n,)
            Fitness values for all frogs in the memplex.

        Returns
        -------
        array-like, shape (sfla_q,) : indices of frogs in the submemplex
        """
        order = np.argsort(fitness_values)[::-1]
        weights = np.zeros(self.sfla_n)
        weights[order] = np.arange(self.sfla_n)
        weights = (2 * (self.sfla_n - weights)
                   / (self.sfla_n * (self.sfla_n + 1)))
        return self._rng.choice(
            self.sfla_n, size=self.sfla_q, replace=False, p=weights)

    def __leap(self, frogs, fitness_values, memplex, submemplex, X, y,
               relief_weights):
        """Create a new worst frog in the submemplex by trying to make it
        leap towards better frogs.

        Parameters
        ----------
        frogs : array-like, shape (sfla_n * sfla_m, n_features)
            The population of frogs.
        fitness_values : array-like, shape (sfla_n * sfla_m)
            Fitness value for each frog.
        memplex : array-like, shape (sfla_n,)
                Indices of frogs in the memplex.
        submemplex : array-like, shape (sfla_q,)
            Indices of frogs in the submemplex.
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        relief_weights : array-like, shape (n_features,)
            Weights for the features.

        Returns
        -------
        tuple (int, array-like, float) : index of the worst frog in the
        submemplex, the new worst frog and its fitness value
        """
        memplex_best_frog = memplex[np.argsort(fitness_values[memplex])[-1]]
        submemplex_worst_frog = submemplex[np.argsort(
            fitness_values[submemplex])[0]]
        getLogger(__name__).info(
            "Trying to evolve frog %s", submemplex_worst_frog)
        worst_frog_fitness = fitness_values[submemplex_worst_frog]

        new_worst_frog = self.__leap_towards(
            frogs[memplex_best_frog], frogs[submemplex_worst_frog], X, y)
        new_worst_frog_fitness = cross_val_score(
            self._estimator, X[:, new_worst_frog], y, cv=self.cv,
            scoring=self.measure_frogs).mean()
        getLogger(__name__).info(
            "After leaping to submemplex best frog: frog = %s, fitness = %d",
            new_worst_frog, new_worst_frog_fitness)
        if new_worst_frog_fitness > worst_frog_fitness:
            return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness

        population_best_frog = np.argsort(fitness_values)[-1]
        new_worst_frog = self.__leap_towards(
            frogs[population_best_frog], frogs[submemplex_worst_frog], X, y)
        new_worst_frog_fitness = cross_val_score(
            self._estimator, X[:, new_worst_frog], y, cv=self.cv,
            scoring=self.measure_frogs).mean()
        getLogger(__name__).info(
            "After leaping to memplex best frog: frog = %s, fitness = %d",
            new_worst_frog, new_worst_frog_fitness)
        if new_worst_frog_fitness > worst_frog_fitness:
            return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness

        new_worst_frog = self.__generate_frog(relief_weights)
        new_worst_frog_fitness = cross_val_score(
            self._estimator, X[:, new_worst_frog], y, cv=self.cv,
            scoring=self.measure_frogs).mean()
        getLogger(__name__).info(
            "Replacing with random frog: frog = %s, fitness = %d",
            new_worst_frog, new_worst_frog_fitness)
        return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness

    def __leap_towards(self, frog_to, frog_from, X, y):
        """Perform a leap from the worse frog to the better frog.

        Parameters
        ----------
        frog_to : array-like, shape (n_features,)
                The better frog (one to leap to).
        frog_from : array-like, shape (n_features,)
            The worse frog (one that leaps).
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        array-like, shape (n_features,) : the new worst frog
        """
        len_to = np.count_nonzero(frog_to)
        len_from = np.count_nonzero(frog_from)
        if len_to > len_from:
            to_add = frog_to & np.logical_not(frog_from)
            weights = su_measure(X[:, to_add], y)
            weights /= np.sum(weights)
            features_num = min(
                min(
                    int(self._rng.random() * (len_to - len_from)),
                    self.s),
                np.count_nonzero(to_add))
            added_features = self._rng.choice(
                np.flatnonzero(to_add), size=features_num, replace=False,
                p=weights)
            new_worst_frog = np.copy(frog_from)
            new_worst_frog[added_features] = True
        else:
            weights = su_measure(X[:, frog_from], y)
            weights /= np.sum(weights)
            features_num = min(
                min(
                    int(self._rng.random() * (len_from - len_to)),
                    self.s),
                len_from - 1)
            left_features = self._rng.choice(
                np.flatnonzero(frog_from), size=len_from - features_num,
                replace=False, p=weights)
            new_worst_frog = np.full(self.n_features_, False)
            new_worst_frog[left_features] = True
        return new_worst_frog

    def __apply_sfla(self, frogs, X, y, relief_weights):
        """Apply the SFLA algorithm to a population of frogs.

        Parameters
        ----------
        frogs : array-like, shape (sfla_n * sfla_m, n_features)
            The population of frogs.
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        relief_weights : array-like, shape (n_features,)
            Weights for the features.

        Returns
        -------
        array-like, shape (sfla_n * sfla_m, n_features) : the evolved population
        of frogs
        """
        fitness_values = np.vectorize(
            lambda frog: cross_val_score(
                self._estimator, X[:, frog], y, cv=self.cv,
                scoring=self.measure_frogs).mean(),
            signature='(1)->()')(frogs)
        getLogger(__name__).info("Fitness values for frogs: %s", fitness_values)
        for _ in range(self.iterations):
            memplexes = self.__create_memplexes()
            for memplex in memplexes:
                for _ in range(self.iterations_leaps):
                    submemplex = memplex[self.__create_submemplex(
                        fitness_values[memplex])]
                    (worst_frog_index, new_worst_frog,
                    new_worst_frog_fitness) = self.__leap(
                        frogs, fitness_values, memplex, submemplex, X, y,
                        relief_weights)
                    frogs[worst_frog_index] = new_worst_frog
                    fitness_values[worst_frog_index] = new_worst_frog_fitness
        return frogs


    def _fit(self, X, y):
        """Fit the model.

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
        self._rng = np.random.default_rng(self.seed)

        #TODO: adding 1 ensures we have no negative probabilities; the paper
        #doesn't say anything about that
        relief_weights = relief_measure(
            X, y, self.relief_iterations, random_state=self.seed) + 1
        relief_weights /= np.sum(relief_weights)
        getLogger(__name__).info("Relief feature weights: %s", relief_weights)

        frogs = np.vectorize(
            lambda _: self.__generate_frog(relief_weights),
            signature='()->(1)')(np.arange(self.sfla_m * self.sfla_n))
        getLogger(__name__).info("Initial frogs population: %s", frogs)
        frogs = np.vectorize(
            lambda frog: self.__iwssr(X[:, frog], y, frog),
            signature='(1)->(1)')(frogs)
        getLogger(__name__).info("Frogs population after IWSSr: %s", frogs)
        frogs = self.__apply_sfla(frogs, X, y, relief_weights)
        getLogger(__name__).info("Frogs population after SFLA: %s", frogs)

        fitness_values = np.vectorize(
            lambda frog: cross_val_score(
                self._estimator, X[:, frog], y, cv=self.cv,
                scoring=self.measure_frogs).mean(),
            signature='(1)->()')(frogs)
        self.best_score_ = np.max(fitness_values)
        self.selected_features_ = frogs[np.argmax(fitness_values)]
