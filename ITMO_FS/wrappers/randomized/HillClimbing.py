import random
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.base import clone

from ...utils import generate_features, BaseWrapper

class HillClimbingWrapper(BaseWrapper):

    def __init__(self, estimator, scorer, seed=42, cv=3):
        self.estimator = estimator
        self.scorer = scorer
        self.seed = seed
        self.cv = cv

    def _fit(self, X, y):
        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self.estimator)

        self._estimator = clone(self.estimator)
        random.seed(self.seed)

        features = generate_features(X)

        self.selected_features_ = [random.choice(features)]
        score = np.mean(cross_val_score(self._estimator, X[:, self.selected_features_], y, cv=self.cv, scoring=make_scorer(self.scorer)))
        old_score = 0
        while score > old_score and len(self.selected_features_) != len(features):
            self.selected_features_ += [random.choice(list(set(features) - set(self.selected_features_)))]
            old_score = score
            score = np.mean(cross_val_score(self._estimator, X[:, self.selected_features_], y, cv=self.cv, scoring=make_scorer(self.scorer)))
        self._estimator.fit()
