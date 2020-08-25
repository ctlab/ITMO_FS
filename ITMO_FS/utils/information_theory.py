from math import log
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

import numpy as np


def conditional_entropy(x_j, y):
    # H(Y|X)
    buf = [[e[1] for e in g] for _, g in groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x_j)


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
    d = defaultdict(int)
    for obj in x:
        d[obj] += 1
    probs = map(lambda z: float(z) / len(x), d.values())
    return -sum(map(elog, probs))
