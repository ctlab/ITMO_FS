import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from ITMO_FS.utils.data_check import *
from ITMO_FS.utils import BaseTransformer
from ITMO_FS.filters.univariate.measures import su_measure, relief_measure


class IWSSr_SFLA(BaseTransformer):

    """
        Performs the IWSSr-SFLA (Incremental Wrapper Subset Selection with replacement and Shuffled Frog Leaping Algorithm).

        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method
        measure_iwssr : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable object / function with signature 
            measure(estimator, X, y) which should return only a single value that would be used in the IWSSr algorithm.
        measure_frogs : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable object / function with signature 
            measure(estimator, X, y) which should return only a single value that would be used to measure the frog's
            fitness value.
        cv : int
            Number of folds in cross-validation.
        relief_iterations : int
            Amount of iterations to do in the Relief algorithm.
        sfla_m : int
            Amount of memplexes in the SFLA algorithm.
        sfla_n : int
            Amount of frogs in each memplex in the SFLA algorithm.
        sfla_q : int
            Amount of frogs in each submemplex in the SFLA algorithm. Should be lower than sfla_n.
        s : int
            Maximum amount of features that can change during one leap in the SFLA algorithm.
        iterations : int
            Total iteration number in the SFLA algorithm.
        iterations_leaps : int
            Total amount of leaps that each memplex would try to do during each iteration.
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
        >>> algo = IWSSr_SFLA(LogisticRegression())
        >>> X, y = make_classification(n_informative=5, n_redundant=8, n_classes=3)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        >>> est.fit()
        >>> X = est.transform(X)
        >>> algo.fit()
        >>> print(algo.selected_features_)
    """
    def __init__(self, estimator, measure_iwssr='accuracy', measure_frogs='accuracy', cv=3, relief_iterations=None, sfla_m=5, 
        sfla_n=20, sfla_q=8, s=3, iterations=20, iterations_leaps=10, seed=42):
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

    def __generate_frogs(self, weights):
        """
            Generates the initial population of frogs.

            Parameters
            ----------
            weights : array-like, shape (n_features)
                Weights for the features.

            Returns
            -------
            array-like, shape (sfla_m * sfla_n, ) : array of frogs; each frog is described by the selected features
        """
        frogs = []

        for i in range(self.sfla_m * self.sfla_n):
            f_num = np.random.randint(self.n_features_) + 1
            features_selected = np.random.choice(self.n_features_, size=f_num, replace=False, p=weights)
            frogs.append(features_selected)

        return frogs

    def __create_memplexes(self):
        """
            Creates sfla_m memplexes.

            Returns
            -------
            array-like, shape (sfla_m, sfla_n)
                The indices of frogs in each of the memplexes.
        """
        memplexes = [[] for i in range(self.sfla_m)]
        perm = np.random.permutation(self.sfla_m * self.sfla_n)
        for i in range(self.sfla_m * self.sfla_n):
            memplexes[int(perm[i] / self.sfla_n)].append(i)
        return np.array(memplexes)


    def __apply_iwssr(self, frogs, X, y):
        """
            Applies the IWSSr algorithm for each of the frogs.

            Parameters
            ----------
            frogs : array-like, shape (sfla_m * sfla_n)
                The array of frogs.
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.

            Returns
            -------
            array-like, shape (sfla_m * sfla_n, ) : array of frogs
        """
        frogs_transformed = []

        for frog in frogs:
            transformed_frog = frog[self.__iwssr(X[:, frog], y)]
            frogs_transformed.append(np.array(transformed_frog))

        return frogs_transformed

    def __iwssr(self, X, y):
        """
            Performs the IWSSr algorithm for given data.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.

            Returns
            -------
            array-like, shape (n_selected_features) : selected features
        """
        weights = su_measure(X, y)
        order = np.argsort(weights)[::-1]
        selected_features = np.array([order[0]])
        best_measure = cross_val_score(self._estimator, X[:, selected_features], y, cv=self.cv,
                                                    scoring=self.measure_iwssr).mean()
        best_iteration_set = selected_features
        for feature in order[1:]:
            for i in range(len(selected_features)):
                iteration_features = np.append(np.delete(selected_features, i), feature)
                iteration_measure = cross_val_score(self._estimator, X[:, iteration_features], y, cv=self.cv,
                                                    scoring=self.measure_iwssr).mean()
                if iteration_measure > best_measure:
                    best_measure = iteration_measure
                    best_iteration_set = iteration_features
            added_feature = np.append(selected_features, feature)
            added_measure = cross_val_score(self._estimator, X[:, added_feature], y, cv=self.cv,
                                                    scoring=self.measure_iwssr).mean()
            if added_measure > best_measure:
                best_measure = added_measure
                best_iteration_set = added_feature
            selected_features = best_iteration_set
        return selected_features

    def __create_submemplex(self, fitness_values):
        """
            Creates a submemplex from the memplex of frogs.

            Parameters
            ----------
            fitness_values : array-like, shape (sfla_n)
                Fitness values for all frogs in the memplex.

            Returns
            -------
            array-like, shape (sfla_q) : indices of frogs in the submemplex
        """
        order = np.argsort(fitness_values)[::-1]
        weights = np.zeros(self.sfla_n)
        for i in range(self.sfla_n):
            weights[order[i]] = 2 * (self.sfla_n - i) / (self.sfla_n * (self.sfla_n + 1))
        return np.random.choice(self.sfla_n, size=self.sfla_q, replace=False, p=weights)

    def __leap(self, frogs, fitness_values, memplex, submemplex, X, y, relief_weights):
        """
            Creates a new worst frog in the submemplex by trying to make it leap towards better frogs.

            Parameters
            ----------
            frogs : array-like, shape (sfla_n * sfla_m, )
                The population of frogs.
            fitness_values : array-like, shape (sfla_n * sfla_m)
                Fitness value for each frog.
            memplex : array-like, shape (sfla_n)
                Indices of frogs in the memplex.
            submemplex : array-like, shape (sfla_q)
                Indices of frogs in the submemplex.
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.
            relief_weights : array-like, shape (n_features)
                Weights for the features.

            Returns
            -------
            tuple (int, array-like, float) : index of the worst frog in the submemplex, the new worst frog and its fitness value
        """
        memplex_best_frog = memplex[np.argsort(fitness_values[memplex])[-1]]
        submemplex_worst_frog = submemplex[np.argsort(fitness_values[submemplex])[0]]
        worst_frog_fitness = fitness_values[submemplex_worst_frog]

        new_worst_frog = self.__leap_towards(frogs[memplex_best_frog], frogs[submemplex_worst_frog], X, y)
        new_worst_frog_fitness = cross_val_score(self._estimator, X[:, new_worst_frog], y, cv=self.cv,
                                                scoring=self.measure_frogs).mean()
        if new_worst_frog_fitness > worst_frog_fitness:
            return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness

        population_best_frog = np.argsort(fitness_values)[-1]
        new_worst_frog = self.__leap_towards(frogs[population_best_frog], frogs[submemplex_worst_frog], X, y)
        new_worst_frog_fitness = cross_val_score(self._estimator, X[:, new_worst_frog], y, cv=self.cv,
                                                scoring=self.measure_frogs).mean()
        if new_worst_frog_fitness > worst_frog_fitness:
            return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness

        new_worst_frog = np.random.choice(self.n_features_, size=np.random.randint(self.n_features_) + 1, 
                                            replace=False, p=relief_weights)
        new_worst_frog_fitness = cross_val_score(self._estimator, X[:, new_worst_frog], y, cv=self.cv,
                                                scoring=self.measure_frogs).mean()
        if new_worst_frog_fitness > worst_frog_fitness:
            return submemplex_worst_frog, new_worst_frog, new_worst_frog_fitness
        else:
            return submemplex_worst_frog, frogs[submemplex_worst_frog], worst_frog_fitness

    def __leap_towards(self, frog_to, frog_from, X, y):
        """
            Performs a leap from the worse frog to the better frog.

            Parameters
            ----------
            frog_to : array-like
                The better frog (one to leap to).
            frog_from : array-like
                The worse frog (one that leaps).
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.

            Returns
            -------
            array-like : the new worst frog
        """
        if len(frog_to) > len(frog_from):
            to_add = np.setdiff1d(frog_to, frog_from)
            weights = su_measure(X[:, to_add], y)
            weights /= np.sum(weights)
            features_num = min(min(int(np.random.rand() * (len(frog_to) - len(frog_from))), self.s), len(to_add))
            new_worst_frog = np.append(frog_from, np.random.choice(to_add, size=features_num, replace=False, p=weights))
        else:
            weights = su_measure(X[:, frog_from], y)
            weights /= np.sum(weights)
            features_num = min(min(int(np.random.rand() * (len(frog_from) - len(frog_to))), self.s), len(frog_from) - 1)
            new_worst_frog = np.random.choice(frog_from, size=len(frog_from) - features_num, replace=False, p=weights)
        return new_worst_frog

    def __apply_sfla(self, frogs, X, y, relief_weights):
        """
            Applies the SFLA algorithm to a population of frogs.

            Parameters
            ----------
            frogs : array-like, shape (sfla_n * sfla_m, )
                The population of frogs.
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.
            relief_weights : array-like, shape (n_features)
                Weights for the features.

            Returns
            -------
            array-like, shape (sfla_n * sfla_m, ) : the evolved population of frogs
        """
        for _ in range(self.iterations):
            fitness_values = np.array(list(map(lambda frog: cross_val_score(self._estimator, X[:, frog], y, cv=self.cv,
                                                    scoring=self.measure_frogs).mean(), frogs)))
            memplexes = self.__create_memplexes()
            for memplex in memplexes:
                for _ in range(self.iterations_leaps):
                    submemplex = memplex[self.__create_submemplex(fitness_values[memplex])]
                    worst_frog_index, new_worst_frog, new_worst_frog_fitness = self.__leap(frogs, fitness_values, memplex, submemplex, X, y, relief_weights)
                    frogs[worst_frog_index] = new_worst_frog
                    fitness_values[worst_frog_index] = new_worst_frog_fitness
        return frogs


    def _fit(self, X, y):
        """
            Fits the model.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.

            Returns
            -------
            None
        """

        np.random.seed(self.seed)
        self._estimator = clone(self.estimator)

        relief_weights = relief_measure(X, y, self.relief_iterations)
        relief_weights /= np.sum(relief_weights)

        frogs = self.__generate_frogs(relief_weights)
        frogs = self.__apply_iwssr(frogs, X, y)
        frogs = self.__apply_sfla(frogs, X, y, relief_weights)

        fitness_values = np.array(list(map(lambda frog: cross_val_score(self._estimator, X[:, frog], y, cv=self.cv,
                                                    scoring=self.measure_frogs).mean(), frogs)))
        self.selected_features_ = frogs[np.argmax(fitness_values)]
