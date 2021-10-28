import unittest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from ITMO_FS.hybrid.HybridRFE import HybridRFE
from random import randint
from numpy.random import permutation


class HybridRFETest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators = [SVC(kernel='linear'),
                           RandomForestClassifier(max_depth=2, random_state=0),
                           GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)]
        self.weighted = [False, True]

        n_samples = 1000
        n_classes = 5
        rr = permutation(n_classes)
        self.data = [[] for _ in range(n_samples)]
        self.target = []
        for i in range(n_samples):
            for j in range(2, 4):
                self.data[i].append(j)
                self.data[i].append(i % j)
            self.data[i].append(int(i / (n_samples / n_classes)))
            self.data[i].append(randint(0, n_samples))
            self.data[i].append(rr[int(i / (n_samples / n_classes))])
            self.data[i].append(int(i / (n_samples / n_classes)) + 1)
            self.target.append(int(i / (n_samples / n_classes)))
        self.support = [False, False, False, False, True, False, True, True]
        self.n_support = 3

    def test(self):
        for w in self.weighted:
            for estimator in self.estimators:
                print('Weighted' if w else 'Simple', ' ', estimator, flush=True)
                hybrid = HybridRFE(estimator, self.n_support, w)
                hybrid = hybrid.fit(self.data, self.target)
                assert self.support == hybrid.get_support()


if __name__ == '__main__':
    unittest.main()
