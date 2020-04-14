import numpy as np
from math import exp
from math import log


def calc_conditional_entropy(x_j, y):
    Kx = max(x_j)
    countX = np.zeros(Kx + 1)
    dictYByX={i+1:{} for i in range(Kx)}
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
        partEntropy = sum(map(lambda inClass : elog(inClass / countX[i]), curDict.values()))
        entropy += countX[i] / len(y) * partEntropy
    return -entropy

def matrix_mutual_information(X, y):
    return np.apply_along_axis(mutual_information, 0, X, y)

def mutual_information(x, y):
    return calc_entropy(y) - calc_conditional_entropy(x, y)

def calc_conditional_mutual_information(x, y, z):
    return calc_entropy(list(zip(x, z))) + calc_entropy(list(zip(y, z))) - calc_entropy(list(zip(x, y, z))) - calc_entropy(z)

def calc_joint_mutual_information(x, y, z):
    return mutual_information(x, z) + calc_mutual_conditional_information(y, z, x)

def calc_interaction_information(x, y, z):
    return mutual_information(x, z) + mutual_information(y, z) - calc_joint_mutual_information(x, y, z)

def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def calc_entropy(x):
	d = dict()
	for obj in x:
		d[obj] = d.get(obj, 0) + 1
	probs = map(lambda z: float(z)/len(x), d.values())
	return -sum(map(elog, probs))    