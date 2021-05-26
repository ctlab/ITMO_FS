import numpy as np
from sklearn.utils import check_array
from ...utils import BaseTransformer, generate_features, apply_cr
from ...filters.univariate.measures import CR_NAMES, MEASURE_NAMES
from sklearn.utils.validation import check_is_fitted


class BestSum(BaseTransformer):
    def __init__(self, models, cutting_rule):
        if len(models) == 0:
            raise ValueError("No models are set")
        self.models = models
        self.cutting_rule = cutting_rule

    def _fit(self, X, y, **kwargs):
        feature_names = generate_features(X)
        self.features = dict(zip(feature_names, np.zeros(len(feature_names))))
        for model in self.models:
            model.fit(X,y)
            for i, k in enumerate(model.selected_features_):
                self.features[k] += ((model.best_score_ - self.features[k]) /
                                     (i + 1))
        self.selected_features_ = apply_cr(self.cutting_rule)(self.features)

    # def predict(self, X):
    #    n = len(self.models)
    #    result = np.zeros((X.shape[0], n))
    #    for i, model in enumerate(self.models):
    #        result[:, i] = model.predict(X[:, self.selected_features_])
    #    return np.array([1 if i else 0 for i in result.sum(axis=1) / n > 0.5])
