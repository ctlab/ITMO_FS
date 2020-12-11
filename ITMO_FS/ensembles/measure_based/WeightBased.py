from collections import defaultdict

from numpy import ndarray, ones
from sklearn.base import clone

from ITMO_FS.utils.data_check import *
from ...utils import BaseTransformer
from .fusion_functions import *


class WeightBased(BaseTransformer):

    def __init__(self, filters, cutting_rule, fusion_function=weight_fusion, weights=None):
        """
        TODO comments
        :param filters:
        """
        self.filters = filters
        self.cutting_rule = cutting_rule
        self.fusion_function = fusion_function
        self.weights = weights

    def get_score(self, X, y):
        self.feature_scores_ = defaultdict(list)
        for __filter in self.filters:
            _filter = clone(__filter)
            _filter.fit(X, y, store_scores=True)
            _min = min(_filter.feature_scores_.values())
            _max = max(_filter.feature_scores_.values())
            for key, value in _filter.feature_scores_.items():
                self.feature_scores_[key].append((value - _min) / (_max - _min))
        return self.feature_scores_

    def __len__(self):
        return len(self.filters)

    def _fit(self, X, y):
        """
        TODO comments
        :param X:
        :param y:
        :param feature_names:
        :return:
        """
        check_filters(self.filters)
        self.feature_scores_ = self.get_score(X, y)
        if self.weights is None:
            weights = ones(len(self.filters)) / len(self.filters)
        else:
            weights = self.weights
        self.selected_features_ = self.cutting_rule(self.fusion_function(self.feature_scores_, weights))
