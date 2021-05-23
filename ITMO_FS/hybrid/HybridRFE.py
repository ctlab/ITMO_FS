import random
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from copy import copy


class HybridRFE:
    """
        Performs the Hybrid-Recursive Feature Elimination.

        Parameters
        ----------
        estimator : Estimator instance
            Model in which the target will be searched.
        n_features_to_select : int, optional (by default half of the n_features)
            The number of selected features.
        weighted : boolean, optional (by default False)
            Using simple sum or weighted sum functions.
        n_cross_validation : int, optional (by default 5)
            The parameter of k-fold cross-validation.
        models : array-like, shape (n_models, ), optional (by default [SVM, RF, GBM])
            Models involved in feature elimination.
        weight_functions : array-like, shape (n_models, ),
                optional (by deafault [coef_, feature_importances_, feature_importances_])
            Functions that return feature weights.

        Examples
        --------
        >>> from ITMO_FS.hybrid.HybridRFE import HybridRFE
        >>> from sklearn.svm import SVC
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> svm = SVC(kernel='linear')
        >>> hybrid = HybridRFE(svm)
        >>> svm.fit(data, target)
        >>> hybrid = hybrid.fit(data, target)
        >>> print(svm.score(data, target))
        >>> print(hybrid.score(data, target))

    """

    def __init__(self, estimator, n_features_to_select=None, weighted=False, n_cross_validation=5, models=None,
                 weight_functions=None):
        self.__estimator = copy(estimator)
        self.__n_features_to_select = n_features_to_select
        self.__weighted = weighted
        self.__n_cross_validation = n_cross_validation
        self.__models = [SVC(kernel='linear'),
                         RandomForestClassifier(max_depth=2, random_state=0),
                         GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)] \
            if models is None else models
        self.__weight_functions = [lambda x: (x.coef_ ** 2).sum(axis=0),
                                   lambda x: x.feature_importances_,
                                   lambda x: x.feature_importances_]\
            if weight_functions is None else weight_functions
        self.__support = []

    def fit(self, X, y):
        """
            Fit the Hybrid-Recursive Feature Elimination model and then the underlying estimator on the selected

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

        """

        if self.__n_features_to_select is None:
            self.__n_features_to_select = int(len(X[0]) / 2)
        Xy = list(zip(X, y))
        random.shuffle(Xy)
        X_split, y_split = zip(*Xy)
        score_best = 0
        features_best = []
        X_split = np.array_split(X_split, self.__n_cross_validation)
        y_split = np.array_split(y_split, self.__n_cross_validation)

        for i in range(self.__n_cross_validation):
            n = len(X_split[0][0])
            Xk = X_split.copy()
            yk = y_split.copy()
            Xk.insert(0, Xk.pop(i))
            yk.insert(0, yk.pop(i))
            Xk.insert(0, [[i for i in range(n)]])
            features = []

            while n > 1:
                for model in self.__models:
                    model.fit(self.__flatten(Xk[2:]), self.__flatten(yk[1:]))
                w = [self.__weight_functions[i](self.__models[i]) for i in range(len(self.__models))]
                if self.__weighted:
                    w = [w[i] * self.__models[i].score(Xk[1], yk[0]) for i in range(len(self.__models))]
                w = np.sum(w, axis=0)
                wns = list(zip(w, range(n)))
                wns.sort()
                _, ns = zip(*wns)
                features.insert(0, Xk[0][0][ns[0]])
                Xk = [[np.delete(Xij, ns[0]) for Xij in Xi] for Xi in Xk]
                n -= 1

            features.insert(0, Xk[0][0][0])
            Xk = np.array_split(np.array_split([Xi[f] for Xi in X for f in
                                                features[:self.__n_features_to_select]], len(X)),
                                self.__n_cross_validation)
            yk = y_split.copy()

            score = 0
            for j in range(self.__n_cross_validation):
                Xk.insert(0, Xk.pop(j))
                yk.insert(0, yk.pop(j))
                self.__estimator.fit(self.__flatten(Xk[1:]), self.__flatten(yk[1:]))
                score += self.__estimator.score(Xk[0], yk[0]) * len(Xk[0])

            if score_best < score:
                score_best = score
                features_best = features

        self.__support = [True for _ in range(len(features_best))]
        for i in range(self.__n_features_to_select, len(features_best)):
            self.__support[features_best[i]] = False

        self.__estimator.fit(self.transform(X), y)

        return self

    def transform(self, X):
        """
            Reduce X to the selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.

            Returns
            ------
            X : array-like, shape (n_samples, n_selected_features)
                The input samples with only the selected features.

        """

        return [np.array(Xi)[self.__support] for Xi in X]

    def fit_transform(self, X, y):
        """
            Fit to data, then transform it.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Input samples.
            y : array-like, shape (n_samples, )
                Target values.

            Returns
            ------
            X_new : array-like, shape (n_samples, n_features)
                Transformed array.

        """

        return self.fit(X, y).transform(X)

    def predict(self, X):
        """
            Reduce X to the selected features and then predict using the underlying estimator.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.

            Returns
            ------
            y : array-like, shape (n_samples, )
                The predicted target values.

        """

        return self.__estimator.predict(X)

    def get_support(self):
        """
            Get a mask, or integer index, of the features selected

            Returns
            ------
            support : array-like, shape (n_features, )

        """

        return self.__support

    def score(self, X, y):
        """
            Reduce X to the selected features and then return the score of the underlying estimator.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            y : array-like, shape (n_samples, )
                The target values.

            Returns
            ------
            score : float

        """

        return self.__estimator.score(self.transform(X), y)

    def __flatten(self, X):
        return [item for sublist in X for item in sublist]
