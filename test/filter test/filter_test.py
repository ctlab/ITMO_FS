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
        filtering = Filter("SpearmanCorr", GLOB_CR["Best by value"](0.9999))
        data, target = load_iris(True)
        res = filtering.run(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

    def test_pearson_mat(self):
        data, target = self.orl['X'], self.orl['Y']
        filtering = Filter("PearsonCorr", GLOB_CR["Best by value"](0.0))
        res = filtering.run(data, target)
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

        ## В Аризоне дана только метрика без отсекающих правил
        ## по сути фильтры у них без отсекающих правил, а во врапперах прописаны сами естиматоры

    def test_add_del(self):
        data, target = self.basehock['X'][:, :100], self.basehock['Y']
        lr = LogisticRegression()
        wrapper = Add_del(lr, f1_score)
        wrapper.run(data, target, silent=True)
        print(wrapper.best_score)
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
