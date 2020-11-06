import time
import unittest
import numpy as np
from collections import defaultdict

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

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

    def test_benching_ensembles(self):
        datasets = [make_classification(n_samples=2000, n_features=20 * i, n_informative=i, n_redundant=5 * i) for i in
                    [2, 10, 20, 50, 100, 200, 500, 1000]]

        filters = [gini_index,
                   fechner_corr,
                   spearman_corr,
                   pearson_corr]

        kfold = KFold(n_splits=10)
        for dataset in datasets:
            X, y = dataset
            k = int(X.shape[1] * 0.1)

            time_ens_start = []
            time_ens_end = []

            time_filter_start = defaultdict(list)
            time_filter_end = defaultdict(list)

            scores_ens = []
            scores_filters = defaultdict(list)
            scores_no_fs = []

            for train_index, test_index in kfold.split(X):
                svm = SVC()
                svm.fit(X[train_index], y[train_index])
                y_pred = svm.predict(X[test_index])
                scores_no_fs.append(f1_score(y[test_index], y_pred))

                time_ens_start.append(time.time())
                ensemble = Mixed(filters)
                ensemble.fit(X[train_index], y[train_index])
                X_transformed = ensemble.transform(X, k, borda_fusion)
                time_ens_end.append(time.time())

                svm = SVC()
                svm.fit(X_transformed[train_index], y[train_index])
                y_pred = svm.predict(X_transformed[test_index])
                scores_ens.append(f1_score(y[test_index], y_pred))

                for filter in filters:
                    time_filter_start[filter.__name__].append(time.time())
                    univ_filter = UnivariateFilter(filter, cutting_rule=("K best", k))
                    univ_filter.fit(X[train_index], y[train_index])
                    X_transformed = univ_filter.transform(X)
                    time_filter_end[filter.__name__].append(time.time())

                    svm = SVC()
                    svm.fit(X_transformed[train_index], y[train_index])
                    y_pred = svm.predict(X_transformed[test_index])
                    scores_filters[filter.__name__].append(f1_score(y[test_index], y_pred))

            print('Dataset size', X.shape)

            sum_time = 0
            for filter in filters:
                filter_dif = np.array(time_filter_end[filter.__name__]) - np.array(time_filter_start[filter.__name__])
                print('Filter ' + filter.__name__ + ' time', np.mean(filter_dif), np.std(filter_dif))
                sum_time += np.mean(filter_dif)

            ens_dif = np.array(time_ens_end) - np.array(time_ens_start)
            print('Ensemble time', np.mean(ens_dif), np.std(ens_dif))
            print('Sum of filter time', sum_time)

            print('No fs score', np.mean(scores_no_fs), np.std(scores_no_fs))

            for filter in filters:
                print('Filter ' + filter.__name__ + ' time', np.mean(scores_filters[filter.__name__]),
                      np.std(scores_filters[filter.__name__]))

            print('Ensemble score', np.mean(scores_ens), np.std(scores_ens))
            print()

if __name__ == '__main__':
    unittest.main()
