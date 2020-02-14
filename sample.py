from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from ITMO_FS.filters import *


informative = 8
# x, y = make_classification(n_samples = 1000, n_features = 20, n_informative = informative, n_redundant = 0)
# for i in range(x.shape[0]):
# 	x[i] = list(map(lambda t: int(t), x[i]))
# print(x)
x = np.array([[1, 2, 3, 2, 2], [2, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
y = np.array([1, 1, 2, 1, 2])
ranks = QPFS().run(x, y)
print(ranks)