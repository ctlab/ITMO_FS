import time
import unittest

import scipy.io
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from ..filters.Filter import *
from ..hybrid.Melif import Melif
from ..wrappers.AddDelWrapper import *
from ..wrappers.BackwardSelection import *




class TestCases(unittest.TestCase):
    basehock = scipy.io.loadmat('BASEHOCK.mat')
    coil = scipy.io.loadmat('COIL20.mat')
    orl = scipy.io.loadmat('orlraws10P.mat')

    # def test_relief(self):
    #     n = 10
    #     x = np.random.randint(n, size=(n, 6))
    #     y = np.random.randint(n, size=n)
    #     # print(y)
    #     print(DefaultMeasures.reliefF_measure(x, y, 6))
    #     # skrebate
    #     R = ReliefF()
    #     R.fit(x, y)
    #     print(R.feature_importances_)

    def test_filter(self):
        data, target = load_iris(True)
        res = Filter("SpearmanCorr", GLOB_CR["Best by value"](0.9999)).run(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

    ##----------Filters------------------------------
    def __compare_measure__(self, measure_name, data):
        data, target = data['X'], data['Y']

        start_time = time.time()
        custom = lambda x, y: np.sum(x + y, axis=1)
        f = Filter(custom, GLOB_CR["K best"](6))
        res = f.run(data, target)  # Filter(measure_name, GLOB_CR["K best"](6)).run(data, target)
        print("ITMO_FS time --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        res = SelectKBest(GLOB_MEASURE[measure_name], k=6).fit_transform(data, target)
        print("SKLEARN time --- %s seconds ---" % (time.time() - start_time))
        print(data.shape, '--->', res.shape)

    def test_pearson_mat(self):
        self.__compare_measure__("PearsonCorr", self.basehock)  # samples:1993 features:4862
        # ITMO    0.12962937355041504 seconds
        # SKLEARN 0.12220430374145508 seconds
        self.__compare_measure__("PearsonCorr", self.coil)  # samples:1440 features:1024
        # ITMO    0.01994633674621582 seconds
        # SKLEARN 0.024933815002441406 seconds
        self.__compare_measure__("PearsonCorr", self.orl)  # samples:100 features:10304
        # ITMO    0.018949508666992188 seconds
        # SKLEARN 0.016954898834228516 seconds

    def test_gini_index_mat(self):
        self.__compare_measure__("GiniIndex", self.basehock)  # samples:1993 features:4862
        # ITMO    0.3725132942199707 seconds
        # SKLEARN 0.3790163993835449 seconds
        self.__compare_measure__("GiniIndex", self.coil)  # samples:1440 features:1024
        # ITMO    0.04388904571533203 seconds
        # SKLEARN 0.056841373443603516 seconds
        self.__compare_measure__("GiniIndex", self.orl)  # samples:100 features:10304
        # ITMO    0.04886913299560547 seconds
        # SKLEARN 0.04886794090270996 seconds

    ##----------Wrapper------------------------------
    def test_backward_selection(self):
        data, target = self.basehock['X'], self.basehock['Y']
        lr = LogisticRegression()
        wrapper = BackwardSelection(lr, 100, GLOB_MEASURE["GiniIndex"])
        wrapper.fit(data[:, :200], target)
        print(wrapper.best_score)
        wrapper.fit(data[:, :300], target)
        print(wrapper.best_score)

    ##----------Melif--------------------------------
    def test_melif(self):
        data, target = self.basehock['X'], self.basehock['Y']
        _filters = [Filter('GiniIndex', cutting_rule=GLOB_CR["Best by value"](0.4)),
                    # Filter('FitCriterion', cutting_rule=GLOB_CR["Best by value"](0.0)),
                    Filter(GLOB_MEASURE["FRatio"](data.shape[1]), cutting_rule=GLOB_CR["Best by value"](0.6)),
                    Filter('InformationGain', cutting_rule=GLOB_CR["Best by value"](-0.4))]
        melif = Melif(_filters, f1_score)
        melif.fit(data, target)
        estimator = SVC()
        melif.run(GLOB_CR['K best'](50), estimator)

    ##----------END----------------------------------
    def test_add_del(self):
        data, target = self.basehock['X'][:, :100], self.basehock['Y']
        lr = LogisticRegression()
        wrapper = Add_del(lr, f1_score)
        wrapper.run(data, target, silent=True)
        print(wrapper.best_score)

    # def test_arizona(self):
    #     data, target = self.coil['X'], self.coil['Y']
    #     start_time = time.time()
    #     features = gini_index.gini_index(data, target)
    #     print("ARIZONA time --- %s seconds ---" % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     features = GLOB_MEASURE["GiniIndex"](data, target)
    #     print("ITMO time --- %s seconds ---" % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     features = f_score.f_score(data, target)
    #     print("ARIZONA time --- %s seconds ---" % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     features = GLOB_MEASURE["FRatio"](data.shape[-1])(data, target)
    #     print("ITMO time --- %s seconds ---" % (time.time() - start_time))

    # @classmethod
    # def __test_mrmr(cls, data, target):
    #     n = data.shape[1] / 2
    #     res = Filter(GLOB_MEASURE["MrmrDiscrete"](n), GLOB_CR["Best by value"](0.0)).run(data, target)
    #     print("Mrmr:", data.shape, '--->', res.shape)
    #
    # def test_mrmr_basehock(self):
    #     MyTestCase.__test_mrmr(self.basehock['X'], self.basehock['Y'])
    #
    # def test_mrmr_coil(self):
    #     MyTestCase.__test_mrmr(self.coil['X'], self.coil['Y'])
    #
    # def test_mrmr_orl(self):
    #     MyTestCase.__test_mrmr(self.orl['X'], self.orl['Y'])


if __name__ == '__main__':
    unittest.main()
