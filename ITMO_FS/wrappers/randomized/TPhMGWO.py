import math

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from ...utils import BaseWrapper


class TPhMGWO(BaseWrapper):
    """
        Performs Grey Wolf optimization with Two-Phase Mutation

        Parameters
        ----------
        wolfNumber : int
            Number of search agents used to find solution for features selection problem
        seed : int
            Random seed used to initialize np.random.seed()
        alpha : float
            weight of importance of classification accuracy
            Note alpha is used in equation that counts fitness as fitness = alpha * score + beta * |selected_features| / |features| where alpha = 1 - beta
        estimator : estimator used for training and testing on provided dataset
            Note that algorithm implementation assumes that estimator has fit, predict methods
            Default algorithm uses sklearn.neighbors.KNeighborsClassifier
        foldNumber : int
            fold number to train and test estimator
        iteration_number : int
            number of iterations of algorithm
        Mp : float
            probability of mutation

        Notes
        -----
        For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0957417419305263/>`_.


        Examples
        --------
        >>> import numpy as np
        >>> from ITMO_FS.wrappers.randomized import TPhMGWO
        >>> from sklearn.datasets import make_classification
        >>> tphmgwo = TPhMGWO()
        >>> x, y = make_classification(500, 50, n_informative = 10, \
n_redundant = 30, n_repeated = 10, shuffle = True)
        >>> tphmgwo.fit()
        >>> print(tphmgwo.selected_features_)


    """

    def __init__(self, wolfNumber=10, seed=1, alpha=0.01, estimator=KNeighborsClassifier(n_neighbors=3), foldNumber=5,
                 iteration_number=30, Mp=0.5, errorRate=mean_squared_error):
        self.wolfNumber = wolfNumber
        self.seed = seed
        self.alpha = alpha
        self.estimator = estimator
        self.foldNumber = foldNumber
        self.iteration_number = iteration_number
        self.errorRate = errorRate
        self.Mp = Mp

    class ClassifierMethodsException(Exception):
        pass

    def __sigmoid(self, x):
        rand = np.random.random()
        xs = np.where(1 / (1 - math.exp(-x)) <= rand, 1, 0)
        return xs

    def __tanh(self, x):
        rand = np.random.random()
        xs = np.where(abs(np.tanh(x)) <= rand, 1, 0)
        return xs

    def __genRandValues(self, t, featureNumber):
        a = 2 - t * 2.0 / self.iteration_number
        rand1 = np.random.uniform(low=0.0, high=1.0, size=featureNumber)
        rand2 = np.random.uniform(low=0.0, high=1.0, size=featureNumber)
        A = abs(2 * a * rand1 - a)
        C = 2 * rand2
        return a, A, C

    def __calcFitness(self, wolves, X,
                      y):  # Calc fitness function. Note that it returns an array and takes an array of search agents
        featureNumber = X.shape[1]
        fitnessFunc = np.zeros(self.wolfNumber, dtype=np.float)
        for index in range(wolves.shape[0]):
            wolf = wolves[index]
            XFilt = X[:, np.where(wolf == 1)[0]]
            kf = KFold(self.foldNumber, shuffle=True)
            score = 0.0
            for train_indices, test_indices in kf.split(X):
                self._estimator.fit()
                score += self.errorRate(y[test_indices], self._estimator.predict(XFilt[test_indices]),
                                             squared=False)
            score /= self.foldNumber
            fitnessFunc[index] = self.alpha * score + self._beta * np.count_nonzero(wolf) / featureNumber
        return fitnessFunc

    def _fit(self, X, y):
        """
            Runs the TPhGWO algorithm on the specified dataset.
            
            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The input samples.
            y : array-like, shape (n_samples)
                The classes for the samples.

            Returns
            ------
            array-like, shape (n_samples,n_selected_features) : 0-1 array where 1 means feature is selected and 0 not

        """

        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self.estimator)
        if not hasattr(self.estimator, 'predict'):
            raise TypeError("estimator should be an estimator implementing "
                            "'predict' method, %r was passed" % self.estimator)
        self._estimator = clone(self.estimator)
        self._beta = 1 - self.alpha


        featureNumber = X.shape[1]
        np.random.seed(self.seed)
        wolves = []
        for i in range(self.wolfNumber):
            wolves.append(np.random.randint(0, 2, featureNumber, dtype = np.integer))
        #print(wolves)
        wolves = np.array(wolves)
        wolves = wolves.astype(dtype=np.double)
        for wolf in wolves:
            if wolf.sum() == 0:
                wolf[0] = 1
        a, A, C = self.__genRandValues(0, featureNumber)
        classNumber = np.unique(y).size
        fitnessFunc = self.__calcFitness(wolves, X, y)
        deltaIndex, betaIndex, alphaIndex = np.argpartition(fitnessFunc, -3)[-3:]
        best3Wolves = [wolves[deltaIndex], wolves[betaIndex], wolves[alphaIndex]]
        for t in range(self.iteration_number):
            for index in range(len(wolves)):
                Darray = abs(C * best3Wolves - wolves[index])
                wolfAprox = best3Wolves - A * Darray
                wolfNewPos = np.mean(wolfAprox, axis=0)
                wolves[index] = wolfNewPos
            wolves = self.__tanh(wolves)
            for wolf in wolves:
                if wolf.sum() == 0:
                    wolf[0] = 1
            a, A, C = self.__genRandValues(t, featureNumber)
            fitnessFunc = self.__calcFitness(wolves, X, y)
            deltaIndex, betaIndex, alphaIndex = np.argpartition(fitnessFunc, -3)[-3:]
            best3Wolves = [np.copy(wolves[deltaIndex]), np.copy(wolves[betaIndex]), np.copy(wolves[alphaIndex])]
            alphaWolf = np.copy(wolves[alphaIndex])
            fitnessFuncAlpha = self.__calcFitness(np.array([alphaWolf]), X, y)[0]
            one_positions = np.where(alphaWolf == 1)[0]
            mutatedAlphaOne = np.copy(alphaWolf)
            for one_pos in one_positions:
                r = np.random.random()
                if r < self.Mp:
                    mutatedAlphaOne[one_pos] = 0
                    if mutatedAlphaOne.sum() == 0:
                        break
                    mutatedFitnessAlpha = self.__calcFitness(np.array([mutatedAlphaOne]), X, y)[0]
                    if (mutatedFitnessAlpha < fitnessFuncAlpha):
                        fitnessFuncAlpha = mutatedFitnessAlpha
                        alphaWolf = mutatedAlphaOne
            # mutatedAlphaOne = np.copy(wolves[alphaIndex])
            zero_positions = np.where(alphaWolf == 0)[0]
            mutatedAlphaZero = np.copy(alphaWolf)
            for zero_pos in zero_positions:
                r = np.random.random()
                if r < self.Mp:
                    mutatedAlphaZero[zero_pos] = 1
                    mutatedFitnessAlpha = self.__calcFitness(np.array([mutatedAlphaZero]), X, y)[0]
                    if (mutatedFitnessAlpha < fitnessFuncAlpha):
                        fitnessFuncAlpha = mutatedFitnessAlpha
                        alphaWolf = mutatedAlphaZero
            # mutatedAlphaZero = np.copy(wolves[alphaIndex])
            wolves[alphaIndex] = alphaWolf
        if alphaWolf.sum() == 0:
            alphaWolf[0] = 1
        self.selected_features_ = np.where(alphaWolf == 1)[0]
        self._estimator.fit()
