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
        self.__filters = filters

    def fit(self, X, y, feature_names=None):
        feature_names = genearate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        self.__feature_names = feature_names
        self.__X = X
        self.__y = y
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters)

    def run(self, cutting_rule=None, estimator=None, test_size=0.3):
        check_cutting_rule(cutting_rule)
        train_x, train_y, test_x, test_y = train_test_split(self.__X, self.__y, test_size)
        t = 1
        nu = {i: [] for i in self.__feature_names}
        for _filter in self.__filters:
            for key, value in _filter.run(train_x, train_y).items():
                nu[key].append(value)

        while t < 100:
            nu = dict(zip(nu.keys(), self.measure(np.array(list(nu.values())))))
            F = cutting_rule(nu, -0.5)
            estimator.fit(np.take(train_x, list(F.keys())), train_y)
            self.gradient_descend(test_y, estimator.predict(test_x))
            t += 1

        pass

    def gradient_descend(self, true, predicted):
        lambd = 0.001
        self.__filter_weights = self.__filter_weights - lambd * 2 * (true - predicted)

    def measure(self, nu):
        return np.dot(nu, self.__filter_weights)
