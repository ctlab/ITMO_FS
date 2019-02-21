import numpy as np

from utils.data_check import *


class Melif:
    __filters = []
    __feature_names = []
    __filter_weights = []
    __scorer = ""
    __y = []
    __X = []
    __alphas = []

    def __init__(self, filters, scorer=""):  ##todo scorer name
        check_filters(filters)
        scorer = check_scorer(scorer)
        self.__classifiers = filters
        self.__scorer = scorer

    def fit(self, X, y, features_names=None):
        if features_names is None:
            features_names = []
        check_shapes(X, y)
        check_features(features_names)
        self.__X = X
        self.__y = y
        self.__feature_names = features_names
        self.__filter_weights = np.ones(len(self.__filters))

    def run(self, cutting_rule="", estimator=None):
        check_cutting_rule(cutting_rule)
        t = 1
        nu = []
        for _filter in self.__filters:
            nu.append(_filter.ran(self.__X))
        while t < 100:
            nu = self.measure(np.array(nu))
            F = cutting_rule(nu)
            estimator.fit(F, self.__y)
            t += 1
        
        pass

    def measure(self, nu):
        return np.dot(self.__filter_weights, nu)
