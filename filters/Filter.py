from functools import partial

import numpy as np

import filters


# TODO: move all feature_names?


# Default measures
class DefaultMeasures:
    FitCriterion = filters.FitCriterion()  # Can be customized
    # TODO: .run() feature_names

    GiniIndex = filters.GiniIndexFilter()

    # IGFilter = filters.IGFilter()  # TODO: unexpected .run() interface; .run() feature_names; no default constructor

    # RandomFilter = filters.RandomFilter() # TODO: bad .run() interface; .run() feature_names; no default constructor

    SpearmanCorrelation = filters.SpearmanCorrelationFilter()

    # SymmetricUncertainty = filters.SymmetricUncertainty()  # TODO

    VDM = filters.VDM()  # TODO: probably not a filter


def spearman_corr(x, y):
    n = x.shape[0]
    c = 6 / (n * (n - 1) * (n + 1))

    dif = x - np.vstack(tuple([y] * x.shape[1])).T
    return 1 - c * np.sum(dif * dif, axis=0)


def pearson_corr(x, y):
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sum_dev = y_dev.dot(x_dev)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    return sum_dev / np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x))




# print(DefaultMeasures.SpearmanCorrelation)

GLOB_MEASURE = {"FitCriterion": DefaultMeasures.FitCriterion, "SpearmanCorr": spearman_corr,
                "PearsonCorr": pearson_corr}

GLOB_CR = {"Best by value": select_best_by_value, "Worst by value": select_worst_by_value}


class DefaultCuttingRules:

    def select_best_by_value(value):
        return partial(_select_by_value, value=value, more=True)

    def select_worst_by_value(value):
        return partial(_select_by_value, value=value, more=False)

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

    def select_k_best(k):
        return partial(self.__select_k, k=k)

    def select_k_worst(k):
        return partial(__select_k, k=-k)

    def __select_k(scores, k):
        return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1])[:k]]


class Filter:
    def __init__(self, measure, cutting_rule):
        self.measure = measure
        self.cutting_rule = cutting_rule
        self.feature_scores = None

    def run(self, x, y, store_scores=False):
        self.feature_scores = None
        feature_scores = self.measure(x, y)
        if store_scores:
            self.feature_scores = feature_scores
        selected_features = self.cutting_rule(feature_scores)
        return x[selected_features]
