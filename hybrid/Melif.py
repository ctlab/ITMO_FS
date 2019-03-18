import datetime as dt
import logging

import numpy as np

from utils.data_check import *

logging.basicConfig(level=logging.INFO)

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
        logging.info('Runing basic MeLiF\nFilters:{}'.format(self.__filters))

    def fit(self, X, y, feature_names=None):
        feature_names = genearate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        self.__feature_names = feature_names
        self.__X = X
        self.__y = y
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters)

    def run(self, cutting_rule=None, estimator=None, test_size=0.3):
        logging.info("Estimator: {}".format(estimator))
        logging.info(
            "Optimizer gradient descent, optimizing measure is Mean squeared error")  # TODO add optimizer and quality measure
        time = dt.datetime.now()
        logging.info("time:{}".format(time))
        check_cutting_rule(cutting_rule)
        train_x, train_y, test_x, test_y = train_test_split(self.__X, self.__y, test_size)
        t = 1
        nu = {i: [] for i in self.__feature_names}
        for _filter in self.__filters:
            for key, value in _filter.run(train_x, train_y).items():
                nu[key].append(value)

        while t < 500:
            time = dt.datetime.now()
            logging.info('Time:{}'.format(dt.datetime.now() - time))
            logging.info('point:{}'.format(self.__filter_weights))
            n = dict(zip(nu.keys(), self.measure(np.array(list(nu.values())))))
            F = cutting_rule(n, 0.0)
            if F == {}:
                break
            estimator.fit(train_x[list(F.keys())], train_y)
            logging.info('Measure at current point : {}'.format(
                np.sum((test_y - estimator.predict(test_x[list(F.keys())])) ** 2)))
            self.gradient_descend(test_y, estimator.predict(test_x[list(F.keys())]))
            t += 1
        logging.info('Footer')
        logging.info("Best point:{}".format(self.__filter_weights))
        logging.info('Top features:')
        for key, value in sorted(F.items(), key=lambda x: x[1], reverse=True):
            logging.info("Feature: {}, value: {}".format(key, value))
        return F

    def gradient_descend(self, true, predicted):
        lambd = 0.1
        self.__filter_weights = self.__filter_weights - lambd * 2 * np.sum(true - predicted) / len(true)

    def measure(self, nu):
        return np.dot(nu, self.__filter_weights)
