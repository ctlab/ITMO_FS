import datetime as dt
import logging

import numpy as np

from utils.data_check import *

logging.basicConfig(level=logging.INFO)


class Melif:
    __filters = []
    __feature_names = []
    __filter_weights = []
    __y = []
    __X = []
    __alphas = []

    def __init__(self, filters, score=None):  ##todo scorer name
        check_filters(filters)
        self.__classifiers = filters
        self.__filters = filters
        self.__score = score

    def fit(self, X, y, feature_names=None, points=None):
        logging.info('Runing basic MeLiF\nFilters:{}'.format(self.__filters))
        feature_names = generate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        self.__feature_names = feature_names
        self.__X = X
        self.__y = y
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters)
        self.points = points

    def run(self, cutting_rule=None, estimator=None, test_size=0.3, delta=0.5):
        self.__estimator = estimator
        self.__cutting_rule = cutting_rule
        self.__delta = delta
        logging.info("Estimator: {}".format(estimator))
        logging.info(
            "Optimizer greedy search, optimizing measure is {}".format(
                self.__score))  # TODO add optimizer and quality measure
        time = dt.datetime.now()
        logging.info("time:{}".format(time))
        check_cutting_rule(cutting_rule)
        self.train_x, self.train_y, self.test_x, self.test_y = train_test_split(self.__X, self.__y, test_size)
        nu = {i: [] for i in self.__feature_names}
        for _filter in self.__filters:
            _filter.run(self.train_x, self.train_y, feature_names=self.__feature_names)
            for key, value in _filter.feature_scores.items():
                nu[key].append(value)
        if self.points is None:
            self.points = [self.__filter_weights]
        best_point = self.points[0]
        best_measure = 0
        best_f = {}
        for point in self.points:
            score, try_point, F = self.search(point, nu)
            if score > best_measure:
                best_measure = score
                best_point = point
                best_f = F
        self.best_measure = best_measure
        self.best_point = best_point
        logging.info('Footer')
        logging.info("Best point:{}".format(best_point))
        logging.info("Best Measure:{}".format(best_measure))
        logging.info('Top features:')
        for key, value in sorted(best_f.items(), key=lambda x: x[1], reverse=True):
            logging.info("Feature: {}, value: {}".format(key, value))
        return best_f

    def search(self, point, features):
        i = 0
        best_point = point
        best_measure = 0
        best_f = {}
        points = [point]
        time = dt.datetime.now()
        while i < len(points):
            point = points[i]
            logging.info('Time:{}'.format(dt.datetime.now() - time))
            logging.info('point:{}'.format(point))
            n = dict(zip(features.keys(), self.measure(np.array(list(features.values())), point)))
            F = self.__cutting_rule(n)
            if F == {}:
                break  # TODO rewrite that thing
            keys = list(F.keys())
            self.__estimator.fit(self.train_x[:, keys], self.train_y)
            predicted = self.__estimator.predict(self.test_x[:, keys])
            score = self.__score(self.test_y, predicted)
            logging.info(
                'Measure at current point : {}'.format(score))
            if score > best_measure:
                best_measure = score
                best_point = point
                best_f = F
                points += self.get_candidates(point, self.__delta)
            i += 1
        return best_measure, best_point, best_f

    def get_candidates(self, point, delta=0.1):
        candidates = np.tile(point, (len(point) * 2, 1)) + np.vstack(
            (np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        return list(candidates)

    def score_features(self, nu, candidate):
        x = np.array(list(nu.values()))
        n = dict(zip(nu.keys(), self.measure()))
        F = self.__cutting_rule(n)
        keys = list(F.keys())
        self.__estimator.fit(self.train_x[list(F.keys())], self.train_y)
        return self.__score(self.test_y, self.__estimator.predict(self.test_x[keys]))

    def measure(self, nu, weights):
        return np.dot(nu, weights)
