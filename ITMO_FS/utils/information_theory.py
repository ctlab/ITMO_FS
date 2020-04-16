from math import log

import numpy as np


def conditional_entropy(x_j, y):
    countX = {x: 0 for x in x_j}
    dictYByX = {x: {} for x in x_j}
    for i in range(len(y)):
        x_val = x_j[i]
        y_val = y[i]
        countX[x_val] += 1
        dictYByX[x_val].update({y_val: dictYByX[x_val].get(y_val, 0) + 1})
    entropy = 0.0
    for x in countX.keys():
        partEntropy = 0.0
        curDict = dictYByX[x]
        partEntropy = sum(map(lambda inClass: elog(inClass / countX[x]), curDict.values()))
        entropy += countX[x] / len(y) * partEntropy
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
    d = dict()
    for obj in x:
        d[obj] = d.get(obj, 0) + 1
    probs = map(lambda z: float(z) / len(x), d.values())
    return -sum(map(elog, probs))
