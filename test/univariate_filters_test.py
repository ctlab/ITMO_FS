import unittest

import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator

from ITMO_FS.filters.univariate import *
from ITMO_FS.filters.univariate.measures import GLOB_CR
from ITMO_FS.utils.information_theory import *
from sklearn.model_selection import KFold

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
        ##TODO Make VDM a Measure and then use it in Univariate filter
        # res = VDM(True).run(data, target)  # Temporary solution
        # assert data.shape[0] == res.shape[0] == res.shape[1]
        # print("Value Difference Metric:", data.shape, '--->', res.shape)
        #
        # data = np.random.randint(10, size=(100, 200))
        # target = np.random.randint(10, size=(100,))
        # res = VDM(True).run(data, target)  # Temporary solution
        # assert data.shape[0] == res.shape[0] == res.shape[1]
        # print("Value Difference Metric:", data.shape, '--->', res.shape)

    def test_k_best(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        for i in [5, 10, 20]:
            res = UnivariateFilter(spearman_corr, select_k_best(i)).fit_transform(data, target)
            assert i == res.shape[1]

    def test_corr(self):
        # Consistency
        data, target = self.wide_classification[0], self.wide_classification[1]
        for f in [spearman_corr, pearson_corr, fechner_corr, kendall_corr]:
            # X.shape == 1 case
            np.testing.assert_array_almost_equal(np.round(f(data[:, 0], data[:, 0]), 10), np.array([1.]))
            # X.shape == 2 case
            res = f(data, target)
            assert (-1 <= res).all() and (1 >= res).all()

    def test_corr_verification(self):
        # Values verification
        data, target = np.array([[1, 8], [3, 2], [4, 5], [0, 7], [7, 1], [8, 4]]), np.array([9, 3, 4, 6, 2, 1])
        for f, verif in zip([spearman_corr, pearson_corr],
                            [stats.spearmanr, stats.pearsonr]):
            true_res = []
            for i in range(data.shape[1]):
                true_res.append(verif(data[:, i], target)[0])
            res = f(data, target)
            # print(res)
            # print(true_res)
            np.testing.assert_allclose(res, true_res)

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
        np.testing.assert_array_equal(df.values, arr)

        # VDM
        # data, target = np.random.randint(10, size=(100, 200)), np.random.randint(10, size=(100,))
        # f = VDM()
        # df = f.run(pd.DataFrame(data), pd.DataFrame(target))
        #
        # f = VDM()
        # arr = f.run(data, target)
        # np.testing.assert_array_equal(df, arr)

    def test_df_feature_names(self):
        # Univariate filter
        data, target = self.wide_classification[0], self.wide_classification[1]
        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(pd.DataFrame(data), pd.DataFrame(target), feature_names=[str(i) + ' column' for i in data.columns])
        df = f.transform(pd.DataFrame(data))

        f = UnivariateFilter(pearson_corr, select_k_best(50))
        f.fit(data, target)
        arr = f.transform(data)
        np.testing.assert_array_equal(df.values, arr)

    def test_chi2(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)
        res = chi2_measure(X, y)
        true_res = chi2(X, y)[0]
        np.testing.assert_allclose(res, true_res)

    def test_anova(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 20))
        X = np.hstack((X, E))

        res = anova(X, y)
        true_res = f_classif(X, y)[0]
        np.testing.assert_allclose(res, true_res)

    def test_modified_t_score(self):
        X = np.array([[5, 1, 3, 2], [4, 2, 2, 1], [3, 3, 4, 1], [2, 2, 3, 1], [1, 1, 5, 2]])
        y = np.array([1, 1, 2, 2, 2])
        scores = modified_t_score(X, y)

        # true_scores was calculated by hand
        true_numerator = np.array([5/2,1/2,3/2,1/6])
        true_denominator = np.sqrt(np.array([1/2,1/2,1/2,7/30]))
        true_modificator = np.array([(sqrt(3)/2)              / ((0              + 5/(2*sqrt(13)) + 0               ) / 3), 
                                     (sqrt(3)/(2*sqrt(7)))    / ((0              + 3/(2*sqrt(91)) + 4/sqrt(21)      ) / 3),
                                     (3*sqrt(3)/(2*sqrt(13))) / ((5/(2*sqrt(13)) + 3/(2*sqrt(91)) + sqrt(3)/sqrt(13)) / 3),
                                     (1/6)                    / ((0              + 4/sqrt(21)     + sqrt(3)/sqrt(13)) / 3)])

        true_scores = true_numerator / true_denominator * true_modificator
        np.testing.assert_allclose(scores, true_scores)

    def test_igain(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)
        res = information_gain(X, y)
        true_res = mutual_info_classif(X, y)
        np.testing.assert_allclose(res, true_res, rtol=0.12)

    def test_mi(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)
        # check invariant
        X_j = X[:, 1]
        eq_1 = entropy(X_j) - conditional_entropy(y, X_j)
        eq_2 = entropy(y) - conditional_entropy(X_j, y)
        eq_3 = entropy(X_j) + entropy(y) - entropy(list(zip(X_j, y)))
        eq_4 = entropy(list(zip(X_j, y))) - conditional_entropy(X_j, y) - conditional_entropy(y, X_j)
        print(eq_1, eq_2)
        np.testing.assert_allclose(eq_1, eq_2)
        np.testing.assert_allclose(eq_2, eq_3)
        np.testing.assert_allclose(eq_3, eq_4)

    def test_gini_index(self):
        X = np.array([[1, 2, 3], [-1, 2, 3], [1, 2, 2], [-1, 1, 1], [1, 2, 1], [-1, 3, 4]])
        y = np.array([1, 2, 1, 3, 1, 1])
        univ_filter = UnivariateFilter('GiniIndex', cutting_rule=('K best', 2))
        univ_filter.fit(X, y)
        # print(univ_filter.feature_scores)
        scores = univ_filter.feature_scores.values()
        assert all(score <= 1 for score in scores) and all(score >= 0 for score in scores)

    def test_single_class(self):
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 2, 2], [1, 1, 1], [1, 2, 1]])
        y = np.array([1, 1, 1, 1, 1])

        univ_filter = UnivariateFilter('FRatio', cutting_rule=('K best', 1))
        univ_filter.fit(X, y)

    def test_fechner(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)
        univ_filter = UnivariateFilter('FechnerCorr', cutting_rule=('K best', 10))
        univ_filter.fit(X, y)
        # print(univ_filter.selected_features)
        # univ_filter = UnivariateFilter('PearsonCorr', cutting_rule=('K best', 10))
        # univ_filter.fit(X, y)
        # print(univ_filter.selected_features)

    def test_def_cr(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)
        univ_filter = UnivariateFilter('FechnerCorr')
        univ_filter.fit(X, y)
        assert univ_filter.selected_features == [0, 2, 3]

    def test_pipeline(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = X.astype(int)

        pipeline = Pipeline([('FS1', UnivariateFilter('FechnerCorr', ('K best', 2)))])
        result = pipeline.fit_transform(X, y)
        assert result.shape[0] == X.shape[0] and result.shape[1] == 2

        p = Pipeline([('FS1', UnivariateFilter('FechnerCorr', ('K best', 2))), ('E1', LogisticRegression())])
        p.fit(X, y)
        assert 0 <= p.score(X, y) <= 1

        p = Pipeline([('FS1', UnivariateFilter('FechnerCorr', ('K best', 3))),
                      ('FS2', UnivariateFilter('PearsonCorr', ('K best', 1)))])

        result = p.fit_transform(X, y)
        assert result.shape[0] == X.shape[0] and result.shape[1] == 1

        p = Pipeline([('FS1', UnivariateFilter('FechnerCorr', ('K best', 3))),
                      ('FS2', UnivariateFilter('PearsonCorr', ('K best', 1))), ('E1', LogisticRegression())])
        p.fit(X, y)
        assert 0 <= p.score(X, y) <= 1

    def test_est(self):
        univ_filter = UnivariateFilter('FechnerCorr', ('K best', 2))

        assert check_estimator(univ_filter)

    def test_qpfs_restrictions(self):
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, GLOB_CR['Best by value'](0.5))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, GLOB_CR['Worst by value'](0.5))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, GLOB_CR['Worst by percentage'](0.5))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, GLOB_CR['Best by percentage'](0.5))

        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, ('Worst by value', 0.5))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, ('Best by value', 0.5))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, ('Worst by percentage', 0.2))
        self.assertRaises(KeyError, UnivariateFilter, qpfs_filter, ('Best by percentage', 0.2))

        UnivariateFilter(qpfs_filter, GLOB_CR['K best'](2))
        UnivariateFilter(qpfs_filter, GLOB_CR['K worst'](2))

        UnivariateFilter(qpfs_filter, ('K best', 2))
        UnivariateFilter(qpfs_filter, ('K worst', 2))

        univ_filter = UnivariateFilter(qpfs_filter, ('K best', 2))
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target

        univ_filter.fit(X, y)

        print(univ_filter.selected_features)

if __name__ == "__main__":
    unittest.main()
