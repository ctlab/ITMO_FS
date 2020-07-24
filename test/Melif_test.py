import unittest

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.filters import *
from ITMO_FS.hybrid.Melif import Melif


class MyTestCase(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)
    filters = [UnivariateFilter(gini_index),
               UnivariateFilter(pearson_corr),
               UnivariateFilter(spearman_corr)]

    estimator = SVC()
    ensemble = WeightBased(filters)

    melif = Melif(ensemble, f1_score, verbose=True)

    def test_wide(self):
        data, target = self.wide_classification[0], self.wide_classification[1]

        train_data, test_data, train_target, test_target = train_test_split(data, target)
        self.melif.fit(train_data, train_target, self.estimator, select_k_best(1500))

        print(f1_score(test_target, self.melif.predict(test_data)))

    def test_wide_pd(self):
        data, target = pd.DataFrame(self.wide_classification[0]), pd.DataFrame(self.wide_classification[1])
        train_data, test_data, train_target, test_target = train_test_split(data, target)
        self.melif.fit(train_data, train_target, self.estimator, select_k_best(1500),
                       feature_names=[str(i) + ' column' for i in data.columns])
        print(f1_score(test_target, self.melif.predict(test_data)))


if __name__ == '__main__':
    unittest.main()
