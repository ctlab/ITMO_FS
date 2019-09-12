from functools import partial

import numpy as np

import filters


# TODO: move all feature_names?


# Default measures
class DefaultMeasures:
    FitCriterion = filters.FitCriterion()  # Can be customized

    # TODO: .run() feature_names

    @staticmethod
    def gini_index(X, y):
        try:
            x = X.values
            y = y.values
        except AttributeError:
            x = X
        cum_x = np.cumsum(x / np.linalg.norm(x, 1, axis=0), axis=0)
        cum_y = np.cumsum(y / np.linalg.norm(y, 1))
        diff_x = (cum_x[1:] - cum_x[:-1])
        diff_y = (cum_y[1:] + cum_y[:-1])
        return np.abs(1 - np.sum(np.multiply(diff_x.T, diff_y).T, axis=0))

    # IGFilter = filters.IGFilter()  # TODO: unexpected .run() interface; .run() feature_names; no default constructor

    # RandomFilter = filters.RandomFilter() # TODO: bad .run() interface; .run() feature_names; no default constructor

    # SymmetricUncertainty = filters.SymmetricUncertainty()  # TODO

    VDM = filters.VDM()  # TODO: probably not a filter

    @staticmethod
    def spearman_corr(X, y):
        n = X.shape[0]
        c = 6 / (n * (n - 1) * (n + 1))
        dif = X - np.vstack(tuple([y] * X.shape[1])).T
        return 1 - c * np.sum(dif * dif, axis=0)

    @staticmethod
    def pearson_corr(X, y):
        x_dev = X - np.mean(X, axis=0)
        y_dev = y - np.mean(y)
        sum_dev = x_dev.T.dot(y_dev)
        sq_dev_x = x_dev * x_dev
        sq_dev_y = y_dev * y_dev
        return sum_dev / np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x))

    # TODO Fehner correlation,concordation coef


# print(DefaultMeasures.SpearmanCorrelation)

GLOB_MEASURE = {"FitCriterion": DefaultMeasures.FitCriterion, "SpearmanCorr": DefaultMeasures.spearman_corr,
                "PearsonCorr": DefaultMeasures.pearson_corr, "GiniIndex": DefaultMeasures.gini_index}


class DefaultCuttingRules:
    @staticmethod
    def select_best_by_value(value):
        return partial(DefaultCuttingRules.__select_by_value, value=value, more=True)

    @staticmethod
    def select_worst_by_value(value):
        return partial(DefaultCuttingRules.__select_by_value, value=value, more=False)

    @staticmethod
    def __select_by_value(scores, value, more=True):
        features = []
        for key, sc_value in scores.items():
            if more:
                if sc_value >= value:
                    features.append(key)
            else:
                if sc_value <= value:
                    features.append(key)
        return features

    @staticmethod
    def select_k_best(k):
        return partial(DefaultCuttingRules.__select_k, k=k, reverse=True)

    @staticmethod
    def select_k_worst(k):
        return partial(DefaultCuttingRules.__select_k, k=k)

    @classmethod
    def __select_k(cls, scores, k, reverse=False):
        if type(k) != int:
            raise TypeError("Number of features should be integer")
        return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


GLOB_CR = {"Best by value": DefaultCuttingRules.select_best_by_value,
           "Worst by value": DefaultCuttingRules.select_worst_by_value,
           "K best": DefaultCuttingRules.select_k_best, "K worst": DefaultCuttingRules.select_k_worst}


class Filter(object):
    def __init__(self, measure, cutting_rule):
        try:
            self.measure = GLOB_MEASURE[measure]
        except KeyError:
            raise KeyError("No %r measure yet" % measure)

        self.cutting_rule = cutting_rule
        self.feature_scores = None

    def run(self, x, y, store_scores=False, feature_names=None):
        self.feature_scores = None
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        feature_scores = dict(zip(feature_names, self.measure(x, y)))
        if store_scores:
            self.feature_scores = feature_scores
        selected_features = self.cutting_rule(feature_scores)
        return x[:, selected_features]
