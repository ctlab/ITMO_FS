import random as rnd
from copy import copy
from importlib import reload

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


class Add_del(object):
    """
        Creates add-del feature wrapper

        Parameters
        ----------
        estimator: object
            A supervised learning estimator with a fit method
        score : callable
            A callable function which will be used to estimate score
        score : boolean
            maximize = True if bigger values are better for score function
        seed: int
            Seed for python random
        best_score : float
            The best score of given metric on the feature combination after add-del procedure

        See Also
        --------
        Lecture about feature selection (ru), p.13 - http://www.ccas.ru/voron/download/Modeling.pdf

        examples
        --------
        >>> from sklearn.metrics import accuracy_score
        >>> from sklearn import datasets,linear_model
        >>> import pandas as pd
        >>> data = datasets.make_classification(n_samples=1000, n_features=20)
        >>> X = np.array(data[0])
        >>> y = np.array(data[1])
        >>> lg = linear_model.LogisticRegression(solver='lbfgs')
        >>> add_del = Add_del(lg, accuracy_score)
        >>> features = add_del.run(X, y)

        >>> from sklearn.metrics import mean_absolute_error
        >>> boston = datasets.load_boston()
        >>> X = pd.DataFrame(boston['data'], columns=boston['feature_names'])
        >>> y = pd.DataFrame(boston['target'])
        >>> lasso = linear_model.Lasso()
        >>> add_del = Add_del(lasso, mean_absolute_error, maximize=False)
        >>> features = add_del.run(X, y)
        >>> features
        ['ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B']

    """

    def __init__(self, estimator, score, maximize=True, seed=42):
        if not hasattr(estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % estimator)
        self._estimator = estimator
        self.score = score
        self.maximize = maximize
        rnd.seed(seed)
        self.best_score = None

    def __add(self, X, y, cv=3, silent=True):

        prev_score = 0
        current_score = 0
        scores = []

        to_append = [i for i in range(X.shape[1])]  # list of features not used in final configuration
        appended = []  # list of features in final configuration

        for feature in to_append:

            appended.append(feature)

            current_score = abs(np.mean(cross_val_score(self._estimator, X[:, appended], y,
                                                        scoring=make_scorer(self.score,
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

    def __del(self, X, y, features, cv=3, silent=True):

        prev_score = abs(np.mean(cross_val_score(self._estimator, X[:, features], y,
                                                 scoring=make_scorer(self.score, greater_is_better=self.maximize),
                                                 cv=cv)))
        current_score = 0
        scores = [prev_score]

        if not silent:
            print('score: {}'.format(prev_score))

        iter_features = copy(features)

        for feature in iter_features:

            features.remove(feature)

            current_score = abs(np.mean(cross_val_score(self._estimator, X[:, features], y,
                                                        scoring=make_scorer(self.score,
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

    def fit(self, X, y, cv=3, silent=True):  ##TODO with fit predict
        """
           Fits wrapper.

           Parameters
           ----------
           X : numpy array or pandas DataFrame, shape (n_samples, n_features)
               The training input samples.
           y : numpy array of pandas Series, shape (n_samples, )
               The target values.
           cv=3 : int
               Number of splits in cross-validation
           silent=True : boolean
               If silent=False then prints all the scores during add-del procedure

           Returns:
           ----------
           features : list
               List of feature after add-del procedure

           See Also
           --------

           examples
           --------
           :param silent:
           :param y:
           :param X:
           :param cv:

       """

        return_feature_names = False

        try:
            import pandas

            if isinstance(X, pandas.DataFrame):
                return_feature_names = True
                columns = np.array(X.columns)
                return_feature_names = True
            else:
                pandas = reload(pandas)
        except ImportError:
            pass

        X = np.array(X)
        y = np.array(y).ravel()

        if not silent:
            print('add trial')
        features = self.__add(X, y, cv, silent)

        if not silent:
            print('del trial')

        features, score = self.__del(X, y, features, cv, silent)
        self.best_score = score

        if return_feature_names:
            features = list(columns[features])

        self._estimator.fit(X[:, features], y)

    def predict(self, X):
        return self._estimator.predict(X)
