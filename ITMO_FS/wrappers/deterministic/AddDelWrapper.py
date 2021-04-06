import random as rnd
from copy import copy
from importlib import reload

import numpy as np
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from ...utils import BaseWrapper


class AddDelWrapper(BaseWrapper):
    """
        Creates add-del feature wrapper

        Parameters
        ----------
        estimator: object
            A supervised learning estimator with a fit method
        scorer : callable
            A callable function which will be used to estimate score
        cv : int
            Number of splits in cross-validation
        maximize : boolean
            maximize = True if bigger values are better for score function
        seed: int
            Seed for python random
        best_score : float
            The best score of given metric on the feature combination after add-del procedure
        silent : boolean
            If silent=False then prints all the scores during add-del procedure

        See Also
        --------
        Lecture about feature selection (ru), p.13 - http://www.ccas.ru/voron/download/Modeling.pdf

        Examples
        --------
        >>> from sklearn.metrics import accuracy_score
        >>> from sklearn import datasets,linear_model
        >>> data = datasets.make_classification(n_samples=1000, n_features=20)
        >>> X = np.array(data[0])
        >>> y = np.array(data[1])
        >>> lg = linear_model.LogisticRegression(solver='lbfgs')
        >>> add_del = AddDelWrapper(lg, accuracy_score)
        >>> add_del.fit()

        >>> from sklearn.metrics import mean_absolute_error
        >>> boston = datasets.load_boston()
        >>> X = boston['data']
        >>> y = boston['target']
        >>> lasso = linear_model.Lasso()
        >>> add_del = AddDelWrapper(lasso, mean_absolute_error, maximize=False)
        >>> add_del.fit()
    """

    def __init__(self, estimator, scorer, cv=3, maximize=True, seed=42, silent=True):
        self.estimator = estimator
        self.scorer = scorer
        self.cv = cv
        self.maximize = maximize
        self.seed = seed
        self.silent = silent

    def __add(self, X, y, cv, silent):

        prev_score = 0
        scores = []

        to_append = [i for i in range(X.shape[1])]  # list of features not used in final configuration
        appended = []  # list of features in final configuration

        for feature in to_append:

            appended.append(feature)

            current_score = abs(np.mean(cross_val_score(self._estimator, X[:, appended], y,
                                                        scoring=make_scorer(self.scorer,
                                                                            greater_is_better=self.maximize),
                                                        cv=cv)))
            scores.append(current_score)

            if not silent:
                print('feature {} (score: {})'.format(feature, current_score))

            if self.maximize == True and current_score <= prev_score:
                appended.pop()

            elif self.maximize == False and current_score > prev_score:
                appended.pop()

            prev_score = current_score

        if not silent:
            if self.maximize:
                print('max score: {}'.format(np.max(scores)))
            elif not self.maximize:
                print('min score: {}'.format(np.min(scores)))

        return appended

    def __del(self, X, y, features, cv, silent):

        prev_score = abs(np.mean(cross_val_score(self._estimator, X[:, features], y,
                                                 scoring=make_scorer(self.scorer, greater_is_better=self.maximize),
                                                 cv=cv)))
        current_score = 0
        scores = [prev_score]
        res_score = 0
        if not silent:
            print('score: {}'.format(prev_score))

        iter_features = copy(features)

        for feature in iter_features:

            if len(features) == 1:
                break

            features.remove(feature)

            current_score = abs(np.mean(cross_val_score(self._estimator, X[:, features], y,
                                                        scoring=make_scorer(self.scorer,
                                                                            greater_is_better=self.maximize),
                                                        cv=cv)))
            scores.append(current_score)

            if not silent:
                print('remove feature {} (score: {})'.format(feature, current_score))

            if self.maximize and prev_score > current_score:
                features.append(feature)

            if not self.maximize and prev_score <= current_score:
                features.append(feature)

            if self.maximize and current_score > prev_score:
                prev_score = current_score

            if not self.maximize and current_score <= prev_score:
                prev_score = current_score

        if self.maximize:
            res_score = np.max(scores)
        elif not self.maximize:
            res_score = np.min(scores)

        if silent == 'False':
            print('score: {}'.format(res_score))

        return features, res_score

    def _fit(self, X, y, silent=True):
        """
           Fits wrapper.

           Parameters
           ----------
           X : numpy array or pandas DataFrame, shape (n_samples, n_features)
               The training input samples.
           y : numpy array of pandas Series, shape (n_samples, )
               The target values.

           Returns:
           ----------
           None

        """

        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self._estimator)
        self._estimator = clone(self.estimator)
        self.best_score_ = 0.0

        if not self.silent:
            print('add trial')
        features = self.__add(X, y, self.cv, self.silent)

        if not self.silent:
            print('del trial')

        features, score = self.__del(X, y, features, self.cv, self.silent)
        self.best_score_ = score
        
        self.selected_features_ = features
        self._estimator.fit()
