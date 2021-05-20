import random
import numpy as np


class HybridRFE:
    """
        Performs the Hybrid-Recursive Feature Elimination.

        Parameters
        ----------
        models : array-like, shape
            Models involved in feature elimination.
        weight_functions : array-like, shape
            Functions that return feature weights.
        score_function : function
            Combines the scores of the models.

        Examples
        --------
        >>> from ITMO_FS.hybrid.HybridRFE import HybridRFE
        >>> from sklearn.svm import SVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> svm = SVC(kernel='linear')
        >>> rf = RandomForestClassifier(max_depth=2, random_state=0)
        >>> gbm = GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)
        >>> models = [svm, rf, gbm]
        >>> hybrid = HybridRFE(models,
        >>>                    [lambda x: (x.coef_ ** 2).sum(axis=0),
        >>>                     lambda x: x.feature_importances_,
        >>>                     lambda x: x.feature_importances_],
        >>>                    lambda x: sum(x) / len(x))
        >>> selected = hybrid.transform(data, target, weighted=True)
        >>> for m in models:
        >>>     m.fit(data, target)
        >>> print([m.score(data, target) for m in models])
        >>> for m in models:
        >>>     m.fit(selected, target)
        >>> print([m.score(selected, target) for m in models])
    """

    def __init__(self, models, weight_functions, score_function):
        self.models = models
        self.weight_functions = weight_functions
        self.score_function = score_function

    def transform(self, X_input, y_input, m=None, weighted=False, k=5, feature_names=None):
        """
            Hybrid-Recursive Feature Elimination.

            Parameters
            ----------
            X_input : array-like, shape (n_features, n_samples)
                The features values.
            y_input : array-like, shape (n_samples, )
                The target values.
            m : int, optional (by default half of the n_features)
                The number of selected features.
            weighted : boolean, optional (by default False)
                Using simple sum or weighted sum functions.
            k : int, optional (by default 5)
                The parameter of k-fold cross-validation.
            feature_names : list of strings, optional

            Returns
            ------
            X dataset sliced with features selected by the algorithm
            and feature names sliced with features selected by the algorithm
            if there is a corresponding parameter
        """

        if m is None:
            m = int(len(X_input[0]) / 2)
        Xy = list(zip(X_input, y_input))
        random.shuffle(Xy)
        X, y = zip(*Xy)
        score_best = 0
        features_best = []
        X = np.array_split(X, k)
        y = np.array_split(y, k)

        for i in range(k):
            n = len(X[0][0])
            Xk = X.copy()
            yk = y.copy()
            Xk.insert(0, Xk.pop(i))
            yk.insert(0, yk.pop(i))
            Xk.insert(0, [[i for i in range(n)]])
            features = []

            while n > 1:
                if weighted:
                    w = self.__step(self.__flatten(Xk[2:]), self.__flatten(yk[1:]), Xk[1], yk[0])
                else:
                    w = self.__step(self.__flatten(Xk[2:]), self.__flatten(yk[1:]))
                wns = list(zip(w, range(n)))
                wns.sort()
                _, ns = zip(*wns)
                features.insert(0, Xk[0][0][ns[0]])
                Xk = [[np.delete(Xij, ns[0]) for Xij in Xi] for Xi in Xk]
                n -= 1

            features.insert(0, Xk[0][0][0])
            Xk = np.array_split(np.array_split([Xi[f] for Xi in X_input for f in features[:m]], len(X_input)), k)
            yk = y.copy()

            score = 0
            for j in range(k):
                Xk.insert(0, Xk.pop(j))
                yk.insert(0, yk.pop(j))
                self.__fit_models(self.__flatten(Xk[1:]), self.__flatten(yk[1:]))
                score += self.score_function([model.score(Xk[0], yk[0]) for model in self.models]) / len(Xk[0])

            if score_best < score:
                score_best = score
                features_best = features

        X_result = np.array_split([Xi[f] for Xi in X_input for f in features_best[:m]], len(X_input))

        if feature_names is not None:
            feature_names_result = [feature_names[f] for f in features_best[:m]]
            return X_result, feature_names_result

        return X_result

    def __step(self, X_training, y_training, X_test=None, y_test=None):
        self.__fit_models(X_training, y_training)
        w = [self.weight_functions[i](self.models[i]) for i in range(len(self.models))]
        if (X_test is not None) and (y_test is not None):
            w = [w[i] * self.models[i].score(X_test, y_test) for i in range(len(self.models))]
        return np.sum(w, axis=0)

    def __flatten(self, X):
        return [item for sublist in X for item in sublist]

    def __fit_models(self, X, y):
        for model in self.models:
            model.fit(X, y)
