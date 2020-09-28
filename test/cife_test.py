import unittest

import numpy as np

from ITMO_FS.ensembles.cife import CIFE
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris, load_diabetes

class TestCases(unittest.TestCase):
    some_classification = make_classification(n_informative=10)

    def test_cife(self):
        data = self.some_classification[0]
        cife = CIFE()
        result = cife.fit(data)
        assert len(result) <= 10

    def test_on_diabetes(self):
        diabetes_dataset = load_diabetes()
        X = diabetes_dataset.data
        cife = CIFE()
        result = cife.fit(X)
        print(result)
        diabetes_answer = [0, 2, 5, 8]
        np.testing.assert_allclose(result, diabetes_answer)

    def test_on_iris(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        cife = CIFE()
        result = cife.fit(X)
        iris_answer = [0, 2]
        np.testing.assert_allclose(result, iris_answer)