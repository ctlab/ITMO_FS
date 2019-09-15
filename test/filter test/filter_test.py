import unittest

import scipy.io
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from filters.Filter import *
from wrappers.AddDelWrapper import *
from wrappers.BackwardSelection import *


# from skfeature import function as sk


class MyTestCase(unittest.TestCase):
    basehock = scipy.io.loadmat('BASEHOCK.mat')
    coil = scipy.io.loadmat('COIL20.mat')
    orl = scipy.io.loadmat('orlraws10P.mat')

    def test_filter(self):
        data, target = load_iris(True)
        res = Filter("SpearmanCorr", GLOB_CR["Best by value"](0.9999)).run(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

    def test_pearson_mat(self):
        data, target = self.orl['X'], self.orl['Y']
        res = Filter("PearsonCorr", GLOB_CR["Best by value"](0.0)).run(data, target)
        print("PearsonCorr:", data.shape, '--->', res.shape)

    def test_gini_index_mat(self):
        data, target = self.basehock['X'], self.basehock['Y']
        # filtering = Filter("GiniIndex", GLOB_CR["K best"](6))
        res = Filter("GiniIndex", GLOB_CR["K best"](6)).run(data, target)
        print("GiniIndex:", data.shape, '--->', res.shape)

    def test_sklearn(self):
        data, target = self.basehock['X'], self.basehock['Y']
        res = SelectKBest(GLOB_MEASURE["GiniIndex"], k=6).fit_transform(data, target)
        print("SkLearn:", data.shape, '--->', res.shape)

    @classmethod
    def __test_mrmr(cls, data, target):
        n = data.shape[1] / 2
        res = Filter(GLOB_MEASURE["MrmrDiscrete"](n), GLOB_CR["Best by value"](0.0)).run(data, target)
        print("Mrmr:", data.shape, '--->', res.shape)

    def test_mrmr_basehock(self):
        MyTestCase.__test_mrmr(self.basehock['X'], self.basehock['Y'])

    def test_mrmr_coil(self):
        MyTestCase.__test_mrmr(self.coil['X'], self.coil['Y'])

    def test_mrmr_orl(self):
        MyTestCase.__test_mrmr(self.orl['X'], self.orl['Y'])

        ## В Аризоне дана только метрика без отсекающих правил
        ## по сути фильтры у них без отсекающих правил, а во врапперах прописаны сами естиматоры

    # def test_add_del(self):
    #     data, target = self.basehock['X'][:, :100], self.basehock['Y']
    #     lr = LogisticRegression()
    #     wrapper = Add_del(lr, f1_score)
    #     wrapper.run(data, target, silent=True)
    #     print(wrapper.best_score)
    #
    # def test_backward_selection(self):
    #     data, target = self.basehock['X'], self.basehock['Y']
    #     lr = LogisticRegression()
    #     wrapper = BackwardSelection(lr, 100, GLOB_MEASURE["GiniIndex"])
    #     wrapper.fit(data[:, :200], target)
    #     print(wrapper.best_score)
    #     wrapper.fit(data[:, :300], target)
    #     print(wrapper.best_score)


if __name__ == '__main__':
    unittest.main()
