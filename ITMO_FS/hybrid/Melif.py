import datetime as dt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.utils.data_check import *
from ITMO_FS.utils import BaseTransformer


class Melif(BaseTransformer):

    def __init__(self, estimator, cutting_rule, filter_ensemble, scorer=None,
                 test_size=0.3,
                 delta=0.5, points=None, verbose=False,
                 seed=42):  # TODO scorer name
        self.estimator = estimator
        self.cutting_rule = cutting_rule
        self.filter_ensemble = filter_ensemble
        self.scorer = scorer
        self.test_size = test_size
        self.delta = delta
        self.points = points
        self.verbose = verbose
        self.seed = seed

    def _fit(self, X, y):
        """

        :param X:
        :param y:
        :param estimator:
        :param cutting_rule:
        :param test_size:
        :param delta:
        :param points:
        :return:
        """
        filter_ensemble = clone(self.filter_ensemble)
        if filter_ensemble is list:
            self.__ensemble = WeightBased(filter_ensemble)
        else:
            self.__ensemble = filter_ensemble
        if self.verbose:
            print(
                'Running basic MeLiF\nEnsemble of :{}'.format(self.__ensemble))

        self.__filter_weights = np.ones(len(self.__ensemble)) / len(
            self.__ensemble)
        self.best_score_ = 0
        self.best_point_ = []
        self.best_f_ = {}
        self.estimator_ = clone(self.estimator)

        if self.verbose:
            print('Estimator: {}'.format(self.estimator_))
            print("Optimizer greedy search, optimizing measure is {}".format(
                self.scorer))
            time = dt.datetime.now()
            print("time:{}".format(time))

        check_cutting_rule(self.cutting_rule)
        self._train_x, self._test_x, self._train_y, self._test_y = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        nu = self.__ensemble.get_score(X, y)

        if self.points is None:
            points = [self.__filter_weights]
            for i in range(len(self.__ensemble)):
                a = np.zeros(len(self.__ensemble))
                a[i] = 1
                points.append(a)
        else:
            points = self.points
        best_point_ = points[0]
        mapping = dict(zip(range(len(nu.keys())), nu.keys()))
        n = dict(zip(nu.keys(),
                     self.__measure(np.array(list(nu.values())), best_point_)))

        self.selected_features_ = self.cutting_rule(n)
        self.best_f_ = {i: nu[i] for i in self.selected_features_}
        for k, v in mapping.items():
            nu[k] = nu.pop(v)
        self.__search(points, nu)
        self.selected_features_ = [mapping[i] for i in self.selected_features_]
        for k in list(self.best_f_.keys()):
            self.best_f_[mapping[k]] = self.best_f_.pop(k)
        if self.verbose:
            print('Footer')
            print("Best point:{}".format(self.best_point_))
            print("Best Score:{}".format(self.best_score_))
            print('Top features:')
            for key, value in sorted(self.best_f_.items(), key=lambda x: x[1],
                                     reverse=True):
                print("Feature: {}, value: {}".format(key, value))

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def __search(self, points, features):
        i = 0
        border = len(points)
        if self.verbose:
            time = dt.datetime.now()
        while i < len(points):
            point = points[i]
            if self.verbose:
                print('Time:{}'.format(dt.datetime.now() - time))
                print('point:{}'.format(point))
            self.__values = np.array(list(features.values()))
            n = dict(
                zip(features.keys(), self.__measure(self.__values, point)))
            self.selected_features_ = self.cutting_rule(n)
            new_features = {i: features[i] for i in self.selected_features_}
            if new_features == {}:
                break  # TODO rewrite that thing
            self.estimator_.fit()
            predicted = self.estimator_.predict(
                self._test_x[:, self.selected_features_])
            score = self.scorer(self._test_y, predicted)
            if self.verbose:
                print('Score at current point : {}'.format(score))
            if score > self.best_score_ or i < border:
                self.best_score_ = score
                self.best_point_ = point
                self.best_f_ = new_features
                points += self.__get_candidates(point, self.delta)
            i += 1

    def __get_candidates(self, point, delta=0.1):
        candidates = np.tile(point, (len(point) * 2, 1)) + np.vstack(
            (np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        return list(candidates)

    def __score_features(self, nu, candidate):
        n = dict(zip(nu.keys(), self.__measure(nu, candidate)))
        scores = self.cutting_rule(n)
        keys = list(scores.keys())
        self.estimator_.fit()
        return self.scorer(self._test_y,
                           self.estimator_.predict(self._test_x[keys]))

    def __measure(self, nu, weights):
        return np.dot(nu, weights)
