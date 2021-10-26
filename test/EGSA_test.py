import unittest
from ITMO_FS.hybrid.EGSA.EGSA import EGSA
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


class EgsaTest(unittest.TestCase):
    @staticmethod
    def test():
        for informative, features, alpha, expected in [(2, 10, 0.8, 0.94), (10, 50, 0.9, 0.9), (7, 22, 1, 0.94)]:
            X, y = make_classification(n_samples=200, n_features=features, random_state=0,
                                       n_informative=informative, n_redundant=0, shuffle=False)

            egsa = EGSA(alpha=alpha, iterations=50)
            trX = egsa.fit_transform(X, y)

            score = KNeighborsClassifier().fit(trX, y).score(trX, y)

            assert (score >= expected)


if __name__ == '__main__':
    unittest.main()
