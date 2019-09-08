import numpy as np


def spearman_corr(x, y):
    # n = x.shape[0]
    # c = 6 / (n * (n - 1) * (n + 1))
    #
    # dif = x - np.vstack(tuple([y] * x.shape[1])).T
    # return 1 - c * np.sum(dif * dif, axis=0)

    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sum_dev = y_dev.dot(x_dev)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    return sum_dev / np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x))


class SpearmanCorrelationFilter(object):
    feature_scores = {}

    ##todo theory and comments
    def __init__(self, cutting_rule=None):
        self.__cutting_rule = cutting_rule

    def run(self, x, y, feature_names=None):
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        # check_features(feature_names, x.shape[1])

        result = spearman_corr(x, y)
        self.feature_scores = dict(zip(feature_names, result))
        if self.__cutting_rule is None:
            return self.feature_scores
        return self.__cutting_rule(self.feature_scores)

    def __repr__(self):
        return "Spearman correlation with rule {}".format(self.__cutting_rule)
