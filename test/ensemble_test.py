import unittest

from sklearn.datasets import make_classification, make_regression

from ITMO_FS.ensembles.measure_based import *
from ITMO_FS.ensembles.ranking_based import *
from ITMO_FS.filters.univariate import *


class MyTestCase(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)

    def test_ranking_based_ensemble(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        filters = [gini_index,
                   fechner_corr,
                   spearman_corr,
                   pearson_corr]
        ensemble = Mixed(filters)
        ensemble.fit(data, target)
        ensemble.transform(data, 100, borda_fusion)
        d = [{'f' + str(i): i for i in range(100)}.items()] * 5
        self.assertEqual(borda_fusion(d, 100), ['f' + str(i) for i in reversed(range(100))])
        ensemble.transform(data, 100)
        self.assertEqual(borda_fusion(d, 100), ['f' + str(i) for i in reversed(range(100))])

    def test_weight_based_ensemble(self):
        data, target = self.wide_classification[0], self.wide_classification[1]
        filters = [UnivariateFilter(gini_index),
                   UnivariateFilter(fechner_corr),
                   UnivariateFilter(spearman_corr),
                   UnivariateFilter(pearson_corr)]
        ensemble = WeightBased(filters)
        ensemble.fit(data, target)

        weights = [0.5, 0.5, 0.5, 0.5]
        ensemble.transform(data, select_k_best(100), weights=weights)


if __name__ == '__main__':
    unittest.main()
