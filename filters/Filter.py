from functools import partial

import numpy as np

import filters


# TODO: move all feature_names?

# x = np.array([[4, 1, 3, 2, 5],
#                       [5, 4, 3, 1, 4],
#                       [5, 2, 3, 0, 5],
#                       [1, 1, 4, 0, 5]])
# y = np.array([2,
#               1,
#               0,
#               0])
# {0: 0.75, 1: 0.75, 2: 0.5, 3: 1.0, 4: 0.75}

# Default measures
class DefaultMeasures:
    FitCriterion = filters.FitCriterion()  # Can be customized

    # TODO: .run() feature_names

    # return array(ratio)
    @staticmethod
    def fc_measure(X, y):
        x = np.asarray(X)  # Converting input data to numpy array
        y = np.asarray(y)

        fc = np.zeros(x.shape[1])  # Array with amounts of correct predictions for each feature

        tokensN = np.max(y) + 1  # Number of different class tokens

        centers = np.empty(tokensN)  # Array with centers of sets of feature values for each class token
        variances = np.empty(tokensN)  # Array with variances of sets of feature values for each class token
        # Each of arrays above will be separately calculated for each feature

        distances = np.empty(tokensN)  # Array with distances between sample's value and each class's center
        # This array will be separately calculated for each feature and each sample

        for feature_index, feature in enumerate(x.T):  # For each feature
            # Initializing utility structures
            class_values = [[] for _ in range(tokensN)]  # Array with lists of feature values for each class token
            for index, value in enumerate(y):  # Filling array
                class_values[value].append(feature[index])
            for token, values in enumerate(class_values):  # For each class token's list of feature values
                tmp_arr = np.array(values)
                centers[token] = np.mean(tmp_arr)
                variances[token] = np.var(tmp_arr)

            # Main calculations
            for sample_index, value in enumerate(feature):  # For each sample value
                for i in range(tokensN):  # For each class token
                    # Here can be raise warnings by 0/0 division. In this case, default results
                    # are interpreted correctly
                    distances[i] = np.abs(value - centers[i]) / variances[i]
                fc[feature_index] += np.argmin(distances) == y[sample_index]

        fc /= y.shape[0]
        return fc

    # return array(index)
    @staticmethod
    def fratio_measure(X, y):
        f_ratios = []
        for feature in X.T:
            Mu = np.mean(feature)
            inter_class = 0.0
            intra_class = 0.0
            y_t = y.T
            for value in np.unique(y_t):
                index_for_this_value = np.where(y_t == value)[0]
                n = np.sum(feature[index_for_this_value])
                mu = np.mean(feature[index_for_this_value])
                var = np.var(feature[index_for_this_value])
                inter_class += n * np.power((mu - Mu), 2)
                intra_class += (n - 1) * var

            f_ratio = inter_class / intra_class
            f_ratios.append(f_ratio)
        f_ratios = np.array(f_ratios)
        # return top n f_ratios
        return np.argpartition(f_ratios, -10)[-10:]

    # return array(ratio)
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
    # @staticmethod
    # def ig_measure(X, y):

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

GLOB_MEASURE = {"FitCriterion": DefaultMeasures.fc_measure,
                "FRatio": DefaultMeasures.fratio_measure,
                "GiniIndex": DefaultMeasures.gini_index,
                "InformationGain": DefaultMeasures.ig_measure,
                "SpearmanCorr": DefaultMeasures.spearman_corr,
                "PearsonCorr": DefaultMeasures.pearson_corr}


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
           "K best": DefaultCuttingRules.select_k_best,
           "K worst": DefaultCuttingRules.select_k_worst}


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
