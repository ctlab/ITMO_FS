import numpy as np
from math import exp
from math import log


def __calc_conditional_entropy(x_j, y):
    Kx = max(x_j)
    countX = np.zeros(Kx + 1)
    dictYByX = {}
    for i in range(Kx):
        dictYByX.update({i + 1: {}})
    for i in range(len(y)):
        x_val = x_j[i]
        y_val = y[i]
        countX[x_val] += 1
        dictYByX[x_val].update({y_val : dictYByX[x_val].get(y_val, 0) + 1})
    entropy = 0.0
    for i in range(Kx + 1):
        if countX[i] == 0:
            continue
        partEntropy = 0.0
        curDict = dictYByX[i]
        for inClass in curDict.values():
            partEntropy += elog(inClass / countX[i])
        entropy += countX[i] / len(y) * partEntropy
    return -entropy

def __mutual_information(x, y):
    return __calc_entropy(y) - __calc_conditional_entropy(x, y)

def __calc_conditional_mutual_information(x, y, z):
    return __calc_entropy(list(zip(x, z))) + __calc_entropy(list(zip(y, z))) - __calc_entropy(list(zip(x, y, z))) - __calc_entropy(z)

def __calc_joint_mutual_information(x, y, z):
    return __mutual_information(x, z) + __calc_mutual_conditional_information(y, z, x)

def __calc_interaction_information(x, y, z):
    return __mutual_information(x, z) + __mutual_information(y, z) - __calc_joint_mutual_information(x, y, z)

def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def __calc_entropy(x):
	d = dict()
	for obj in x:
		d[obj] = d.get(obj, 0) + 1
	probs = map(lambda z: float(z)/len(x), d.values())
	return -sum(map(elog, probs))    