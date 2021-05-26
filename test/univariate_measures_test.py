import unittest
from ITMO_FS.filters.univariate.measures import *
from utils import load_dataset


class MyTestCase(unittest.TestCase):
    madelon = load_dataset("madelon.csv")

    def test_measures(self):
        data = self.madelon.drop(['target'], axis=1).values
        for f, answer in zip(
                [su_measure,
                 chi2_measure,
                 laplacian_score,
                 information_gain],
                [1,
                 1,
                 1,
                 1]):
            np.testing.assert_allclose(f(data[0].reshape((-1, 1)), data[0]),
                                       answer, atol=1e-05)

    def test_gini_index(self):
        data = self.madelon.drop(['target'], axis=1).values
        assert gini_index(data[0].reshape((-1, 1)), data[0]), 0
        with self.assertRaises(ValueError):
            gini_index(data[0, :1], data[0])

    def test_relief(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        with self.assertRaises(ValueError):
            relief_measure(data, target[target > 0])

    def test_relief(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        relief_measure(data, target)

    def test_laplacian_score(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        laplacian_score(data, target)

    def test_cutting_rules(self):
        data = dict(
            zip(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        assert select_k_best(5)(data), ['f10', 'f9', 'f8', 'f7', 'f6']
        assert select_k_worst(5)(data), ['f1', 'f2', 'f3', 'f4', 'f5']

        with self.assertRaises(TypeError):
            select_k_best(0.5)(data)

        assert select_best_by_value(5)(data), ['f10', 'f9', 'f8', 'f7', 'f6']
        assert select_worst_by_value(5)(data), ['f1', 'f2', 'f3', 'f4', 'f5']
        assert select_best_percentage(0.5)(data), ['f10', 'f9', 'f8', 'f7',
                                                   'f6']
        assert select_worst_percentage(0.5)(data), ['f1', 'f2', 'f3', 'f4',
                                                    'f5']

    def test_fit_criterion(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        fit_criterion_measure(data, target)

    def test_anova(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        anova(data, target)

    def test_modified_t_score(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        modified_t_score(data, target)

    def test_f_ratio(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        f_ratio_measure(data, target)

    def test_kendall(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        kendall_corr(data, target)

    def test_fechner(self):
        data, target = (self.madelon.drop(['target'], axis=1).values,
                        self.madelon["target"].values)
        fechner_corr(data, target)


if __name__ == '__main__':
    unittest.main()
