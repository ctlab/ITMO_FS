import numpy as np


class BestSum:  ## TODO refactor

    def _init_(self, models, cutting_rule):
        self.models = models
        self.cutting_rule = cutting_rule
        self.features = None

    def fit(self, x, y, feature_names=None):
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        self.features = dict(zip(feature_names, np.zeros(len(feature_names))))
        for model in self.models:
            model.fit(x, y)
            for i, k in enumerate(model.selected_features):
                self.features[k] += (model.best_score - self.features[k]) / (i + 1)

    def cut(self, cutting_rule=None):
        if cutting_rule is None:
            return self.cutting_rule(self.features)
        return cutting_rule(self.features)

    def predict(self, X):
        new = self.cut(self.features)
        n = len(self.models)
        result = np.zeros((X.shape[0], n))
        for i, model in enumerate(self.models):
            result[:, i] = model.predict(X[:, new])
        return np.array([1 if i else 0 for i in result.sum(axis=1) / n > 0.5])
