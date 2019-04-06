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

    def fit(self, X, y, feature_names=None, score=None):
        feature_names = genearate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        self.__feature_names = feature_names
        self.__X = X
        self.__y = y
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters)
        self.__score = score

    def run(self, cutting_rule=None, estimator=None, test_size=0.3):
        self.__estimator = estimator
        self.__cutting_rule = cutting_rule
        logging.info("Estimator: {}".format(estimator))
        logging.info(
            "Optimizer gradient descent, optimizing measure is {}".format(
                self.__score))  # TODO add optimizer and quality measure
        time = dt.datetime.now()
        logging.info("time:{}".format(time))
        check_cutting_rule(cutting_rule)
        self.train_x, self.train_y, self.test_x, self.test_y = train_test_split(self.__X, self.__y, test_size)

        t = 1
        nu = {i: [] for i in self.__feature_names}
        for _filter in self.__filters:
            for key, value in _filter.run(self.train_x, self.train_y).items():
                nu[key].append(value)

        points = [self.__filter_weights]
        best_point = self.__filter_weights
        best_measure = 0
        best_F = {}
        for point in points:
            time = dt.datetime.now()
            logging.info('Time:{}'.format(dt.datetime.now() - time))
            logging.info('point:{}'.format(point))
            n = dict(zip(nu.keys(), self.measure(np.array(list(nu.values())))))
            F = cutting_rule(n, 0.0)
            if F == {}:
                break
            keys = list(F.keys())
            estimator.fit(self.train_x[list(F.keys())], self.train_y)
            predicted = estimator.predict(self.test_x[keys])
            score = self.__score(self.test_y, predicted)
            logging.info(
                'Measure at current point : {}'.format(score))
            if score > best_measure:
                best_measure = score
                best_point = point
                best_F = F
                points += self.get_candidates(point)

        logging.info('Footer')
        logging.info("Best point:{}".format(best_point))
        logging.info('Top features:')
        for key, value in sorted(best_F.items(), key=lambda x: x[1], reverse=True):
            logging.info("Feature: {}, value: {}".format(key, value))
        return best_F

    def get_candidates(self, point, delta=0.5):
        candidates = np.tile(point, (len(point) * 2, 1)) + np.vstack(
            (np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        return candidates

    def score_features(self, nu, candidate):
        x = np.array(list(nu.values()))
        n = dict(zip(nu.keys(), self.measure()))
        F = self.__cutting_rule(n, 0.0)
        keys = list(F.keys())
        self.__estimator.fit(self.train_x[list(F.keys())], self.train_y)
        return self.__score(self.test_y, self.__estimator.predict(self.test_x[keys]))

    def measure(self, nu):
        return np.dot(nu, self.__filter_weights)
