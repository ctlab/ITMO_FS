import unittest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression

from ITMO_FS.filters.univariate import *

np.random.seed(42)


# get manually counted values
def fechner_example(a, b):
    if 'used' not in vars(fechner_example).keys():
        fechner_example.used = True
        return [-2 / 3]
    return [2 / 3]


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
        # Consistency
        data, target = self.wide_classification[0], self.wide_classification[1]
        for f in [spearman_corr, pearson_corr, fechner_corr, ]:
            # X.shape == 1 case
            assert f(data[:, 0], data[:, 0]) == 1
            # X.shape == 2 case
            res = f(data, target)
            assert (-1 <= res).all() and (1 >= res).all()

        # Values verification
        data, target = np.array([[1, 8], [3, 2], [4, 5], [0, 7], [7, 1], [8, 4]]), np.array([9, 3, 4, 6, 2, 1])
        for f, verif in zip([fechner_corr, spearman_corr, pearson_corr, ],
                            [fechner_example, stats.spearmanr, stats.pearsonr, ]):
            true_res = []
            for i in range(data.shape[1]):
                true_res.append(verif(data[:, i], target)[0])
            res = f(data, target)
            # print(res)
            # print(true_res)
            assert np.testing.assert_allclose(res, true_res)

    # def test_filters(self):
    #     data, target = self.wide_classification[0], self.wide_classification[1]
    #     for f, answer in zip(
    #             [fit_criterion_measure, f_ratio_measure, gini_index, su_measure, chi2_measure, laplacian_score,
    #              information_gain],
    #             [np.ones((data.shape[1],)), np.ones((data.shape[1],)), np.ones((data.shape[1],)),
    #              np.ones((data.shape[1],)), np.ones((data.shape[1],)), np.ones((data.shape[1],)),
    #              np.ones((data.shape[1],))]):
    #         assert (f(data[0], data[0]) == answer).all()

    def test_df(self):
        # Univariate filter
        data, target = self.wide_classification[0], self.wide_classification[1]
        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(pd.DataFrame(data), pd.DataFrame(target))
        df = f.transform(pd.DataFrame(data))

        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(data, target)
        arr = f.transform(data)
        np.testing.assert_array_equal(df, arr)

        # VDM
        data, target = np.random.randint(10, size=(100, 200)), np.random.randint(10, size=(100,))
        f = VDM()
        df = f.run(pd.DataFrame(data), pd.DataFrame(target))

        f = VDM()
        arr = f.run(data, target)
        np.testing.assert_array_equal(df, arr)


if __name__ == "__main__":
    unittest.main()
