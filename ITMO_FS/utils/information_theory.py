from collections import Counter
from itertools import groupby
from math import log, fsum
from operator import itemgetter

import numpy as np


def conditional_entropy(x_j, y):
    # H(Y|X)
    buf = [[e[1] for e in g] for _, g in
           groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x_j)


def joint_entropy(x, y, z):
    # H(X; Y; Z)
    return entropy(list(zip(x, y, z)))


def matrix_mutual_information(x, y):
    return np.apply_along_axis(mutual_information, 0, x, y)


def mutual_information(x, y):
    # I(X;Y) = H(Y) - H(Y|X)
    return entropy(y) - conditional_entropy(x, y)


def conditional_mutual_information(x, y, z):
    # I(X;Y|Z) = H(X; Z) + H(Y; Z) - H(X; Y; Z) - H(Z)
    return (entropy(list(zip(x, z))) +
            entropy(list(zip(y, z))) -
            entropy(list(zip(x, y, z))) -
            entropy(z))


def joint_mutual_information(x, y, z):
    # I(X, Y; Z) = I(X; Z) + I(Y; Z|X)
    return mutual_information(x, z) + conditional_mutual_information(y, z, x)


def interaction_information(x, y, z):
    return mutual_information(x, z) + mutual_information(y,
                                                         z) - joint_mutual_information(
        x, y, z)


def symmetrical_relevance(x, y, z):
    # SR(X; Y)
    return joint_mutual_information(x, y, z) / joint_entropy(x, y, z)


def elog(x):
    return x * log(x) if 0. < x < 1. else 0.


def entropy(x):
    # H(X)
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)
