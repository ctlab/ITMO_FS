import numpy as np


def estimate_index(X, y):
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



class GiniIndexFilter(object):
    __border = 0.5
    feature_scores = {}

    def __init__(self):
        pass

    # TODO return filtered X
    def run(self, x, y, feature_names=None):
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        # check_features(feature_names, x.shape[1])
        result = estimate_index(x, y)

        self.feature_scores = dict(zip(feature_names, result))
        return dict([i for i in self.feature_scores.items() if i[1] > self.__border])

    def __repr__(self):
        return "Gini Index filter with border {}".format(self.__border)
