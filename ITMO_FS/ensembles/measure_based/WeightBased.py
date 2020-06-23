from collections import defaultdict

from numpy import ndarray, ones
from sklearn.base import TransformerMixin

from ITMO_FS.utils.data_check import *
from .fusion_functions import *


class WeightBased(TransformerMixin):
    __filters = []

    def __init__(self, filters):
        """
        TODO comments
        :param filters:
        """
        check_filters(filters)
        self.__filters = filters
        self.selected_features = None
        self.feature_scores = None

    def __len__(self):
        return len(self.__filters)

    def score(self, X, y, feature_names=None):
        self.feature_scores = defaultdict(list)
        for _filter in self.__filters:
            _filter.fit(X, y, feature_names=feature_names, store_scores=True)
            _min = min(_filter.feature_scores.values())
            _max = max(_filter.feature_scores.values())
            for key, value in _filter.feature_scores.items():
                self.feature_scores[key].append((value - _min) / (_max - _min))
        return self.feature_scores

    def fit(self, X, y, feature_names=None):
        """
        TODO comments
        :param X:
        :param y:
        :param feature_names:
        :return:
        """
        feature_names = generate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        _feature_names = feature_names
        _X = X
        _y = y
        self.feature_scores = self.score(_X, _y, _feature_names)

    def transform(self, x, cutting_rule, fusion_function=weight_fusion, weights=None):
        """
        Transfrom dataset
        :param x:
        :param cutting_rule:
        :param fusion_function:
        :param weights:
        :return:
        """
        if weights is None:
            weights = ones(len(self.__filters)) / len(self.__filters)
        self.selected_features = cutting_rule(fusion_function(self.feature_scores, weights))
        if type(x) is ndarray:
            return x[:, self.selected_features]
        else:
            return x[self.selected_features]

    def fit_transform(self, X, y=None, **fit_params):
        """
        TODO comments
        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        cutting_rule, feature_names, fusion_function, weights = fit_params
        self.fit(X, y, feature_names)
        return self.transform(X, cutting_rule, fusion_function, weights)

    def __repr__(self):
        result = 'Filter weight based ensemble with: \n'
        for f in self.__filters:
            result += str(f) + ' filter \n'
        return result
