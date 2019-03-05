import numpy as np

# from utils.data_check import *


def normalize(array):
    return (array - min(array)) / (max(array) - min(array))


class SpearmanCorrelation:
    __border = 0.5
    __features = {}

    ##todo theory and comments
    def __init__(self, border=0.5):
        self.__border = border

    def run(self, x, y, feature_names=None):
        if feature_names is None:
            feature_names = list(range(x.shape[1]))
        # check_features(feature_names, x.shape[1])
        x_dev = x - np.mean(x, axis=0)
        y_dev = y - np.mean(y)
        sum_dev = y_dev.dot(x_dev)
        sq_dev_x = x_dev ** 2
        sq_dev_y = y_dev ** 2
        result = normalize(sum_dev / np.sqrt(sq_dev_y.dot(sq_dev_x)))
        self.__features = dict(zip(feature_names, result))
        return dict([i for i in self.__features.items() if i[1] > self.__border])
