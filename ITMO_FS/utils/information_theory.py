from math import fsum, log
from collections import Counter, defaultdict

import numpy as np


def builder_dict():
    return defaultdict(int)

def conditional_entropy(x_j, y):
    # H(Y|X)
    count_x = defaultdict(int)
    dict_y_by_x = defaultdict(builder_dict)
    for i in range(len(y)):
        x_val = x_j[i]
        y_val = y[i]
        count_x[x_val] += 1
        dict_y_by_x[x_val][y_val] += 1
    entropy = 0.0
    for x_key in count_x.keys():
        cur_dict = dict_y_by_x[x_key]
        part_entropy = sum(map(lambda num_y: elog(num_y / count_x[x_key]), cur_dict.values()))
        entropy += count_x[x_key] / len(x_j) * part_entropy
    return -entropy


def matrix_mutual_information(x, y):
    return np.apply_along_axis(mutual_information, 0, x, y)


def mutual_information(x, y):
    return entropy(y) - conditional_entropy(x, y)


def conditional_mutual_information(x, y, z):
    return entropy(list(zip(x, z))) + entropy(list(zip(y, z))) - entropy(list(zip(x, y, z))) - entropy(z)


def joint_mutual_information(x, y, z):
    return mutual_information(x, z) + conditional_mutual_information(y, z, x)


def interaction_information(x, y, z):
    return mutual_information(x, z) + mutual_information(y, z) - joint_mutual_information(x, y, z)


def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x * log(x)


def entropy(x):
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)
