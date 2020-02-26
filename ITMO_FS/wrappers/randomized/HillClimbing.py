import random

from sklearn.model_selection import cross_validate


class HillClimbingWrapper:
    def __init__(self, estimator, scorer):
        self._scorer = scorer
        self._estimator = estimator
        self.features = None

    def fit(self, x, y, feature_names=None, cv=3):
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))

        features = [random.choice(feature_names)]
        score = cross_validate(self._estimator, x[:, features], y, cv=cv, scoring=self._scorer)
        old_score = 0
        while score > old_score:
            features += [random.choice(set(feature_names) - set(features))]
            old_score = score
            score = cross_validate(self._estimator, x[:, features], y, cv=cv, scoring=self._scorer)
        self.features = features

    def predict(self, X):
        return self._estimator.predict(X[:, self.features])
