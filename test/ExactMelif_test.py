import unittest

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ITMO_FS.ensembles import WeightBased
from ITMO_FS.hybrid.Melif import Melif
from ITMO_FS.filters import *

from cpu_melif import ExactMelif


class ExactMelifTest(unittest.TestCase):
    def test_compare(self):
        random_state = 42
        measures = [pearson_corr, spearman_corr, anova]
        kappa = 5

        univariate_filters = list(map(lambda m: UnivariateFilter(m), measures))

        estimator = SVC(random_state=random_state)
        ensemble = WeightBased(univariate_filters)

        melif = Melif(ensemble, f1_score)
        exact_melif = ExactMelif(measures, kappa, estimator, 'f1_macro')

        X, y = make_classification(n_samples=100, n_classes=2, n_features=30, n_informative=kappa)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        melif.fit(X_train, y_train, estimator, select_k_best(kappa))
        print(f'MeLiF: {f1_score(y_test, melif.predict(X_test))}')

        exact_melif.fit(X_train, y_train)
        print(f'Exact MeLiF: {f1_score(y_test, exact_melif.predict(X_test))}')

    # def test_simple(self):
    #     def measure_1(f, _):
    #         return {1: .1, 11: .2, 21: .3, 31: .5, 41: .4, 51: .8, 61: .6, 71: .9, 81: .95, 91: 1}[f[0]]
    #
    #     def measure_2(f, _):
    #         return {1: .2, 11: .3, 21: .1, 31: .5, 41: .8, 51: .4, 61: .6, 71: .9, 81: 1, 91: .95}[f[0]]
    #
    #     def measure_3(f, _):
    #         return {1: .3, 11: .1, 21: .2, 31: .8, 41: .6, 51: .4, 61: .6, 71: .1, 81: .9, 91: .95}[f[0]]
    #
    #     measures = [measure_1, measure_2, measure_3]
    #     kappa = 4
    #     classifier = _MockClassifier(lambda X: [])
    #     quality = 'f1-macro'
    #     X = np.array([
    #         [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
    #         [2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
    #         [3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
    #         [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    #         [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    #         [6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
    #         [7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
    #         [8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
    #         [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    #         [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #     ])
    #     y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #
    #     melif = ExactMelif(measures, kappa, classifier, quality)
    #     melif.select([], [])


class _MockClassifier:
    def __init__(self, predict_f):
        self._predict_f = predict_f

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self._predict_f(X)


if __name__ == '__main__':
    unittest.main()
