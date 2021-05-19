import random
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def hybrid_rfe(X_input, y_input, m=None, weighted=True, k=5, feature_names=None):
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
        weighted : boolean, optional (by default True)
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

    if weighted:
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
                w = __step(__flatten(Xk[2:]), __flatten(yk[1:]), weighted, Xk[1], yk[0])
                wns = list(zip(w, range(n)))
                wns.sort()
                _, ns = zip(*wns)
                features.insert(0, Xk[0][0][ns[0]])
                Xk = [[np.delete(Xij, ns[0]) for Xij in Xi] for Xi in Xk]
                n -= 1

            features.insert(0, Xk[0][0][0])
            Xk = []
            yk = __flatten(y)

            for Xi in X:
                for Xij in Xi:
                    Xk.append([])
                    for f in features[:int(len(Xij) / 2)]:
                        Xk[-1].append(Xij[f])

            svm = SVC(kernel='linear')
            rf = RandomForestClassifier(max_depth=2, random_state=0)
            gbm = GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)
            svm.fit(Xk, yk)
            rf.fit(Xk, yk)
            gbm.fit(Xk, yk)
            score = (svm.score(Xk, yk) + rf.score(Xk, yk) + gbm.score(Xk, yk)) / 3

            if score_best < score:
                score_best = score
                features_best = features

    else:
        n = len(X[0])
        X = list(X)
        X.insert(0, [i for i in range(n)])
        features = []

        while n > 1:
            w = __step(X[1:], y, weighted)
            wns = list(zip(w, range(n)))
            wns.sort()
            _, ns = zip(*wns)
            features.insert(0, X[0][ns[0]])
            X = [np.delete(Xi, ns[0]) for Xi in X]
            n -= 1

        features.insert(0, X[0][0])
        features_best = features

    X_result = []
    for Xi in X_input:
        X_result.append([])
        for f in features_best[:m]:
            X_result[-1].append(Xi[f])

    if feature_names is not None:
        feature_names_result = []
        for f in features_best[:m]:
            feature_names_result.append(feature_names[f])

        return X_result, feature_names_result

    return X_result


def __step(X_training, y_training, weighted, X_test=[], y_test=[]):
    w = {}

    svm = SVC(kernel='linear')
    svm.fit(X_training, y_training)
    w['svm'] = (svm.coef_ ** 2).sum(axis=0)

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_training, y_training)
    w['rf'] = rf.feature_importances_

    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gbm.fit(X_training, y_training)
    w['gbm'] = gbm.feature_importances_

    for key in w:
        w[key] = [(wi - min(w[key])) / (max(w[key]) - min(w[key])) for wi in w[key]]

    if weighted:
        w['svm'] *= np.array(svm.score(X_test, y_test))
        w['rf'] *= np.array(rf.score(X_test, y_test))
        w['gbm'] *= np.array(gbm.score(X_test, y_test))

    return [sum(wi) for wi in zip(*w.values())]


def __flatten(X):
    return [item for sublist in X for item in sublist]
