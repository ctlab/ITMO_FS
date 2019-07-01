import numpy as np


def cartesian(rw, cl):  # returns cartesian product for passed numpy arrays as two paired numpy array
    tmp = np.array(np.meshgrid(rw, cl)).T.reshape(len(rw) * len(cl), 2)
    return tmp.T[0], tmp.T[1]
