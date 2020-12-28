import datetime
import unittest

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from ITMO_FS.ensembles import WeightBased
from ITMO_FS.filters import *
from ITMO_FS.hybrid.Melif import Melif
from ITMO_FS.utils import test_scorer


class MyTestCase(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)
    filters = [UnivariateFilter(gini_index),
               UnivariateFilter(pearson_corr),
               UnivariateFilter(spearman_corr)]

    estimator = SVC(random_state=42)
    ensemble = WeightBased(filters)

    melif = Melif(estimator, select_k_best(1500), ensemble, scorer=f1_score, verbose=True)



    def test_wide(self):
        data, target = self.wide_classification[0], self.wide_classification[1]

        train_data, test_data, train_target, test_target = train_test_split(data, target)
        self.melif.fit(train_data, train_target)

        print(f1_score(test_target, self.melif.predict(test_data)))

    def test_wide_pd(self):
        data, target = pd.DataFrame(self.wide_classification[0]), pd.DataFrame(self.wide_classification[1])
        train_data, test_data, train_target, test_target = train_test_split(data, target)
        self.melif.fit(train_data, train_target)
        print(f1_score(test_target, self.melif.predict(test_data)))

    def test_R(self):
        data = pd.read_csv('C:\\Users\\SomaC\\PycharmProjects\\machinka\\mlrcheck\\boston_corrected.csv')
        target = 'class'
        features = data.loc[:, data.columns != 'b'].columns
        # data[target]=data[target].apply(lambda x: 0 if x<=0 else 1)
        ks = [int(i * 500) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
        print()
        for j in ks:
            print('|' + str(j) + '|')
            start = datetime.datetime.now()
            f = UnivariateFilter(pearson_corr, select_k_best(j))
            f.fit(data[features], data[target])
            print('|', datetime.datetime.now() - start, '|')
            start = datetime.datetime.now()
            f = UnivariateFilter(spearman_corr, select_k_best(j))
            f.fit(data[features], data[target])
            print('|', datetime.datetime.now() - start, '|')
            # start = datetime.datetime.now()
            # f = UnivariateFilter(chi2_measure, select_k_best(j))
            # f.fit(data[features], data[target])
            # print('|', datetime.datetime.now() - start, '|')
            start = datetime.datetime.now()
            f = UnivariateFilter(information_gain, select_k_best(j))
            f.fit(data[features], data[target])
            print('|', datetime.datetime.now() - start, '|')

    def test_est(self):
        melif = Melif(self.estimator, select_k_best(2), self.ensemble, scorer=test_scorer)
        check_estimator(melif)


if __name__ == '__main__':
    unittest.main()
