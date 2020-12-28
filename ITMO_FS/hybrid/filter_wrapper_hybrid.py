from ..utils import BaseWrapper
from sklearn.base import clone

class FilterWrapperHybrid(BaseWrapper):

    def __init__(self, filter_, wrapper):
        self.filter_ = filter_
        self.wrapper = wrapper

    def _fit(self, X, y):
        self._filter = clone(self.filter_)
        self._estimator = clone(self.wrapper)
        new = self._filter.fit_transform(X, y)
        self.selected_features_ = self._filter.selected_features_
        self._estimator.fit(new, y)
        #self.best_score = self.wrapper.best_score

