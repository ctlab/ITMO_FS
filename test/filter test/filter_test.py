import unittest

import scipy.io
from sklearn.datasets import load_iris

from filters.Filter import *


class MyTestCase(unittest.TestCase):
    basehock = scipy.io.loadmat('BASEHOCK.mat')

    def test_filter(self):
        filtering = Filter(GLOB_MEASURE["SpearmanCorr"], GLOB_CR["Best by value"](0.9999))
        data, target = load_iris(True)
        res = filtering.run(data, target)
        print(data.shape, '--->', res.shape)

    def test_pearson_mat(self):
        data, target = self.basehock['X'], self.basehock['Y']
        filtering = Filter(GLOB_MEASURE["PearsonCorr"], GLOB_CR["Best by value"](0.0))
        res = filtering.run(data, target)
        print(data.shape, '--->', res.shape)

    def test_gini_index_mat(self):
        data, target = self.basehock['X'], self.basehock['Y']
        filtering = Filter(GLOB_MEASURE["GiniIndex"], GLOB_CR["K best"](6))
        res = filtering.run(data, target)
        print(data.shape, '--->', res.shape)


if __name__ == '__main__':
    unittest.main()
