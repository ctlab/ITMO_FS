import datetime as dt
import logging

import numpy as np

from ITMO_FS.utils.data_check import *

logging.basicConfig(level=logging.INFO)


class Melif:
    __filters = []
    __feature_names = []
    __filter_weights = []
    __y = []
    __X = []
    __alphas = []
    __estimator = None
    __points = []
    __cutting_rule = None
    __delta = None
    _train_x = _train_y = _test_x = _test_y = None

    def __init__(self, filters, score=None):  # TODO scorer name
        check_filters(filters)
        self.__filters = filters
        self.__score = score
        self.best_score = 0
        self.best_point = []

    def fit(self, X, y, feature_names=None, points=None):
        logging.info('Runing basic MeLiF\nFilters:{}'.format(self.__filters))
        feature_names = generate_features(X, feature_names)
        check_shapes(X, y)
        # check_features(features_names)
        self.__feature_names = feature_names
        self.__X = X
        self.__y = y
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters)
        self.__points = points

    def run(self, cutting_rule, estimator, test_size=0.3, delta=0.5):
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
        self._train_x, self._train_y, self._test_x, self._test_y = train_test_split(self.__X, self.__y, test_size)
        nu = {i: [] for i in self.__feature_names}
        for _filter in self.__filters:
            _filter.run(self._train_x, self._train_y, feature_names=self.__feature_names, store_scores=True)
            for key, value in _filter.feature_scores.items():
                _filter.feature_scores[key] = abs(value)
            _min = min(_filter.feature_scores.values())
            _max = max(_filter.feature_scores.values())
            for key, value in _filter.feature_scores.items():
                nu[key].append((value - _min) / (_max - _min))
        if self.__points is None:
            self.__points = [self.__filter_weights]
        best_point = self.__points[0]
        best_score = 0
        best_f = {}
        for point in self.__points:
            score, try_point, f_dict = self.__search(point, nu)
            if score > best_score:
                best_score = score
                best_point = point
                best_f = f_dict
        self.best_score = best_score
        self.best_point = best_point
        logging.info('Footer')
        logging.info("Best point:{}".format(best_point))
        logging.info("Best Score:{}".format(best_score))
        logging.info('Top features:')
        for key, value in sorted(best_f.items(), key=lambda x: x[1], reverse=True):
            logging.info("Feature: {}, value: {}".format(key, value))
        return best_f

    def __search(self, point, features):
        i = 0
        best_point = point
        best_score = 0
        best_f = {}
        points = [point]
        time = dt.datetime.now()
        while i < len(points):
            point = points[i]
            logging.info('Time:{}'.format(dt.datetime.now() - time))
            logging.info('point:{}'.format(point))
            values = list(features.values())
            n = dict(zip(features.keys(), self.__measure(np.array(values), point)))
            keys = self.__cutting_rule(n)
            new_features = {i: features[i] for i in keys}
            if new_features == {}:
                break  # TODO rewrite that thing
            self.__estimator.fit(self._train_x[:, keys], self._train_y)
            predicted = self.__estimator.predict(self._test_x[:, keys])
            score = self.__score(self._test_y, predicted)
            logging.info(
                'Score at current point : {}'.format(score))
            if score > best_score:
                best_score = score
                best_point = point
                best_f = new_features
                points += self.__get_candidates(point, self.__delta)
            i += 1
        return best_score, best_point, best_f

    @staticmethod
    def __get_candidates(point, delta=0.1):
        candidates = np.tile(point, (len(point) * 2, 1)) + np.vstack(
            (np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        return list(candidates)

    def __score_features(self, nu, candidate):
        n = dict(zip(nu.keys(), self.__measure(nu, candidate)))
        scores = self.__cutting_rule(n)
        keys = list(scores.keys())
        self.__estimator.fit(self._train_x[list(scores.keys())], self._train_y)
        return self.__score(self._test_y, self.__estimator.predict(self._test_x[keys]))

    @staticmethod
    def __measure(nu, weights):
        return np.dot(nu, weights)
