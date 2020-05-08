import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression

from ITMO_FS.filters.univariate import *


class TestCases(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)

    def test_filters(self):
        data, target = load_iris(True)

        res = UnivariateFilter(spearman_corr, select_best_by_value(0.9999)).fit_transform(data, target)
        print("SpearmanCorr:", data.shape, '--->', res.shape)

        data = np.array([[0, 0], [0, 0], [0, 0]])
        target = np.array([3, 3, 3])
        res = VDM(True).run(data, target)  # Temporary solution
        assert data.shape[0] == res.shape[0] == res.shape[1]
        print("Value Difference Metric:", data.shape, '--->', res.shape)

        data = np.random.randint(10, size=(100, 200))
        target = np.random.randint(10, size=(100,))
        res = VDM(True).run(data, target)  # Temporary solution
        assert data.shape[0] == res.shape[0] == res.shape[1]
        print("Value Difference Metric:", data.shape, '--->', res.shape)

    def test_k_best(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for i in [5, 10, 20]:
            res = UnivariateFilter(spearman_corr, select_k_best(i)).fit_transform(data, target)
            assert i == res.shape[1]

    def test_corr(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for f in [fechner_corr, spearman_corr, pearson_corr, ]:
            # X.shape == 1 case
            assert f(data[:, 0], data[:, 0]) == 1
            # X.shape == 2 case
            res = f(data, target)
            assert (-1 <= res).all() and (1 >= res).all()

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


if __name__ == "__main__":
    unittest.main()
