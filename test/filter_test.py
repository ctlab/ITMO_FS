import unittest

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression

from ITMO_FS.filters.UnivariateFilter import *


class TestCases(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)

    def test_filter(self):
        data, target = load_iris(True)
        res = UnivariateFilter(spearman_corr, select_best_by_value(0.9999)).run(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

    def test_k_best(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for i in [5, 10, 20]:
            res = UnivariateFilter(spearman_corr, select_k_best(i)).run(data, target)
            assert i == res.shape[1]

    def test_corr(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for f, answer in zip([spearman_corr, pearson_corr, fechner_corr],
                             [np.ones((data.shape[1],)), np.ones((data.shape[1],)), np.nan]):
            assert (f(data[0], data[0]) == answer).all()
            res = UnivariateFilter(f, select_k_best(5)).run(data, target)
