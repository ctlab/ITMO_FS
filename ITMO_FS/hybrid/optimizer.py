import datetime as dt
import logging

import numpy as np


class GreedOptimizer:
    def __init__(self, features, cutting_rule, estimator, starting_point, delta=0.5):
        self.features = features
        self.cutting_rule = cutting_rule
        self.estimator = estimator
        self.__delta = delta
        self.__starting_point = starting_point

    def __search(self, train_x, train_y, test_x, test_y):
        i = 0
        points = [self.__starting_point]
        time = dt.datetime.now()
        while i < len(points):
            point = points[i]
            logging.info('Time:{}'.format(dt.datetime.now() - time))
            logging.info('point:{}'.format(point))
            self.__values = np.array(list(features.values()))
            n = dict(zip(features.keys(), self.__measure(self.__values, point)))
            self.selected_features = self.cutting_rule(n)
            new_features = {i: features[i] for i in self.selected_features}
            if new_features == {}:
                break  # TODO rewrite that thing
            self.estimator.fit(self._train_x[:, self.selected_features], self._train_y)
            predicted = self.estimator.predict(self._test_x[:, self.selected_features])
            score = self.__score(self._test_y, predicted)
            logging.info('Score at current point : {}'.format(score))
            if score > self.best_score:
                self.best_score = score
                self.best_point = point
                self.best_f = new_features
                points += self.__get_candidates(point, self.__delta)
            i += 1

    def __get_candidates(self, point, delta=0.1):
        candidates = np.tile(point, (len(point) * 2, 1)) + np.vstack(
            (np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        return list(candidates)

    def __score_features(self, nu, candidate):
        n = dict(zip(nu.keys(), self.__measure(nu, candidate)))
        scores = self.__cutting_rule(n)
        keys = list(scores.keys())
        self.__estimator.fit(self._train_x[list(scores.keys())], self._train_y)
        return self.__score(self._test_y, self.__estimator.predict(self._test_x[keys]))

    def __measure(self, nu, weights):
        return np.dot(nu, weights)
