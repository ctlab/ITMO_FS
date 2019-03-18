import numpy as np


def spearmen_corr(x, y):
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
    __border = 0.5
    __features = {}

    ##todo theory and comments
    def __init__(self, border=0.5):
        self.__border = border

    def run(self, x, y, feature_names=None):
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        # check_features(feature_names, x.shape[1])
        result = spearmen_corr(x, y)
        self.__features = dict(zip(feature_names, result))
        return dict([i for i in self.__features.items() if i[1] > self.__border])

    def __repr__(self):
        return "Spearman correlation with border {}".format(self.__border)
