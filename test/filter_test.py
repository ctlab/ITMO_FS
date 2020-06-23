import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ITMO_FS.filters.univariate import *


# from ITMO_FS.filters.univariate import *


class TestCases(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)

    def test_filter(self):
        data, target = load_iris(True)
        res = UnivariateFilter(spearman_corr, select_best_by_value(0.9999)).fit_transform(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

    def test_k_best(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for i in [5, 10, 20]:
            res = UnivariateFilter(spearman_corr, select_k_best(i)).fit_transform(data, target)
            assert i == res.shape[1]

    def test_corr(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for f in [spearman_corr, pearson_corr, fechner_corr]:
            assert (f(data[0], data[0]) == np.ones(data.shape[1])).all()
            # res = UnivariateFilter(f, select_k_best(5)).fit_transform(data, target)

    # def test_filters(self):
    #     data, target = self.wide_classification[0], self.wide_classification[1]
    #     for f, answer in zip(
    #             [fit_criterion_measure, f_ratio_measure, gini_index, su_measure, chi2_measure, laplacian_score,
    #              information_gain],
    #             [np.ones((data.shape[1],)), np.ones((data.shape[1],)), np.ones((data.shape[1],)),
    #              np.ones((data.shape[1],)), np.ones((data.shape[1],)), np.ones((data.shape[1],)),
    #              np.ones((data.shape[1],))]):
    #         assert (f(data[0], data[0]) == answer).all()
    #
    def test_df(self):
        data, target = pd.DataFrame(self.wide_classification[0]), pd.DataFrame(self.wide_classification[1])
        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(data, target)
        df = f.transform(data)
        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(self.wide_classification[0], self.wide_classification[1])
        arr = f.transform(data)
        self.assertEqual(df, arr)
