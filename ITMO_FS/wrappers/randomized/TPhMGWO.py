from logging import getLogger

import numpy as np
from sklearn.model_selection import cross_val_score

from ...utils import BaseWrapper, generate_features


class TPhMGWO(BaseWrapper):
    """Grey Wolf optimization with Two-Phase Mutation.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that should have a fit(X, y) method and
        a predict(X) method. The original paper suggests to use the
        k-nearest-neighbors classifier.
    measure : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable with
        signature measure(estimator, X, y) which should return only a single
        value.
    wolf_number : int
        Number of search agents used to find a solution for feature selection
        problem.
    seed : int
        Random seed used to initialize np.random.default_rng().
    alpha : float
        Weight of importance of classification accuracy. Alpha is used in
        equation that counts fitness as
        fitness = alpha * score + beta * |selected_features| / |features|
        where alpha = 1 - beta.
    cv : int
        Number of folds in cross-validation.
    iteration_number : int
        Number of iterations of the algorithm.
    mp : float
        Mutation probability.
    binarize : str
        Transformation function to use. Currently only 'tanh' and 'sigmoid'
        are supported.

    Notes
    -----
    For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0957417419305263/>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from ITMO_FS.wrappers.randomized import TPhMGWO
    >>> from sklearn.datasets import make_classification
    >>> dataset = make_classification(n_samples=100, n_features=20,
    ... n_informative=5, n_redundant=0, shuffle=False, random_state=42)
    >>> x, y = np.array(dataset[0]), np.array(dataset[1])
    >>> tphmgwo = TPhMGWO(KNeighborsClassifier(n_neighbors=7),
    ... measure='accuracy').fit(x, y)
    >>> tphmgwo.selected_features_
    array([0, 1, 2, 4], dtype=int64)
    """
    def __init__(self, estimator, measure, wolf_number=10, seed=1, alpha=0.5,
                 cv=5, iteration_number=30, mp=0.5, binarize='sigmoid'):
        self.estimator = estimator
        self.measure = measure
        self.wolf_number = wolf_number
        self.seed = seed
        self.alpha = alpha
        self.estimator = estimator
        self.cv = cv
        self.iteration_number = iteration_number
        self.mp = mp
        self.binarize = binarize

    def __sigmoid(self, x):
        rand = self._rng.random(self.n_features_)
        xs = np.where(1 / (1 + np.exp(-x)) <= rand, 1, 0)
        return xs

    def __tanh(self, x):
        rand = self._rng.random(self.n_features_)
        xs = np.where(abs(np.tanh(x)) <= rand, 1, 0)
        return xs

    def __genRandValues(self, t):
        a = 2 - t * 2.0 / self.iteration_number
        rand1 = self._rng.random((3, self.n_features_))
        rand2 = self._rng.random((3, self.n_features_))
        A = abs(2 * a * rand1 - a)
        C = (2 * rand2)
        return A, C

    def _fit(self, X, y):
        """Run the TPhMGWO algorithm on the specified dataset.
            
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
        def __calcScore(wolf):
            score = 1 - cross_val_score(
                self._estimator, X[:, np.where(wolf, True, False)], y,
                cv=self.cv, scoring=self.measure).mean()
            return (self.alpha * score
                    + self._beta * np.count_nonzero(wolf) / self.n_features_)

        def __updateWolf(wolf):
            D = abs(C * best3Wolves - wolf)
            return binarize(np.mean(best3Wolves - A * D, axis=0))

        if self.binarize == 'tanh':
            binarize = self.__tanh
        elif self.binarize == 'sigmoid':
            binarize = self.__sigmoid
        else:
            getLogger(__name__).error(
                "binarize should be 'tanh' or 'sigmoid', %s was passed",
                self.binarize)
            raise ValueError(
                "binarize should be 'tanh' or 'sigmoid', %s was passed"
                % self.binarize)

        self._beta = 1 - self.alpha
        self._rng = np.random.default_rng(self.seed)
        features = generate_features(X)

        wolves = self._rng.choice([0, 1], (self.wolf_number, self.n_features_))
        for wolf in wolves:
            if wolf.sum() == 0:
                wolf[0] = 1
        getLogger(__name__).info("Initial wolves population: %s", wolves)

        A, C = self.__genRandValues(0)
        getLogger(__name__).info("Initial values: A = %s, C = %s", A, C)
        classNumber = np.unique(y).shape[0]

        scores = np.vectorize(
            lambda wolf: __calcScore(wolf), signature='(1)->()')(wolves)
        getLogger(__name__).info("Scores for all wolves: %s", scores)

        alphaIndex, betaIndex, deltaIndex = np.argsort(scores)[:3]
        best3Wolves = wolves[[alphaIndex, betaIndex, deltaIndex]]

        for t in range(self.iteration_number):
            wolves = np.vectorize(
                lambda wolf: __updateWolf(wolf), signature='(1)->(1)')(wolves)
            for wolf in wolves:
                if wolf.sum() == 0:
                    wolf[0] = 1
            getLogger(__name__).info(
                "Wolves population after iteration %d: %s", t, wolves)

            A, C = self.__genRandValues(t)
            getLogger(__name__).info("A = %s, C = %s", A, C)
            scores = np.vectorize(
                lambda wolf: __calcScore(wolf), signature='(1)->()')(wolves)
            getLogger(__name__).info("Scores for all wolves: %s", scores)
            alphaIndex, betaIndex, deltaIndex = np.argsort(scores)[:3]
            best3Wolves = wolves[[alphaIndex, betaIndex, deltaIndex]]

            alphaWolf = np.copy(wolves[alphaIndex])
            alphaScore = scores[alphaIndex]
            mutated = np.copy(alphaWolf)

            getLogger(__name__).info("Base alpha wolf score: %d", alphaScore)
            for selected in self._rng.permuted(np.flatnonzero(alphaWolf)):
                getLogger(__name__).info(
                    "Trying to remove feature %d from the alpha wolf", selected)
                r = self._rng.random()
                if r < self.mp:
                    mutated[selected] = 0
                    if mutated.sum() == 0:
                        break
                    mutatedScore = __calcScore(mutated)
                    getLogger(__name__).info("New score: %d", mutatedScore)
                    if (mutatedScore < alphaScore):
                        alphaScore = mutatedScore
                        alphaWolf = mutated
                    else:
                        mutated[selected] = 1

            mutated = np.copy(alphaWolf)
            for free in self._rng.permuted(np.flatnonzero(alphaWolf - 1)):
                getLogger(__name__).info(
                    "Trying to add feature %d to the alpha wolf", free)
                r = self._rng.random()
                if r < self.mp:
                    mutated[free] = 1
                    mutatedScore = __calcScore(mutated)
                    getLogger(__name__).info("New score: %d", mutatedScore)
                    if (mutatedScore < alphaScore):
                        alphaScore = mutatedScore
                        alphaWolf = mutated
                    else:
                        mutated[free] = 0

            wolves[alphaIndex] = alphaWolf

        if alphaWolf.sum() == 0:
            alphaWolf[0] = 1
        self.selected_features_ = np.flatnonzero(alphaWolf)
        self.best_score_ = cross_val_score(
            self._estimator, X[:, self.selected_features_], y, cv=self.cv,
            scoring=self.measure).mean()
        self._estimator.fit(X[:, self.selected_features_], y)
