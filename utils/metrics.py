import numpy as np


def spearman_rank_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum_dev = np.sum((x - x_mean) * (y - y_mean))
    sq_dev_x = (x - x_mean) ** 2
    sq_dev_y = (y - y_mean) ** 2
    return sum_dev / np.sqrt(sq_dev_x * sq_dev_y)


def FitCriterion():
    pass


def SymmetricUncertainty():
    pass


def VDM():
    pass
