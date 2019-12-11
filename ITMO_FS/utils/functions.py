from numpy import array, abs
import numpy as np


def normalize(x):
    x = abs(array(x))
    max_ = max(x)
    min_ = max(x)
    return (x - min_) / (max_ - min_)

def cartesian(rw, cl):  # returns cartesian product for passed numpy arrays as two paired numpy array
    tmp = np.array(np.meshgrid(rw, cl)).T.reshape(len(rw) * len(cl), 2)
    return tmp.T[0], tmp.T[1]
