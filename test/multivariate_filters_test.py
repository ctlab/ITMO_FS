import unittest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression

from ITMO_FS.filters.multivariate import *

np.random.seed(42)


class TestCases(unittest.TestCase):
    data, target = np.random.randint(10, size=(100, 20)), np.random.randint(10, size=(100,))

    def test_filters(self):
        # FCBF
        res = FCBFDiscreteFilter().run(self.data, self.target)  # Temporary solution
        assert self.data.shape[0] == res.shape[0]
        print("Fast Correlation Based filter:", self.data.shape, '--->', res.shape)

        # DISR
        res = DISRWithMassive().run(self.data, self.target)  # Temporary solution
        assert self.data.shape[0] == res.shape[0]
        print("Double Input Symmetric Relevance:", self.data.shape, '--->', res.shape)

        # TraceRatioFisher
        res = self.data[:, TraceRatioFisher(10).run(self.data, self.target)[0]]  # Temporary solution
        assert self.data.shape[0] == res.shape[0]
        print("TraceRatio:", self.data.shape, '--->', res.shape)

        # Multivariate
        f = MultivariateFilter(MIM, 10)
        f.fit(self.data, self.target)
        res = f.transform(self.data)
        assert self.data.shape[0] == res.shape[0]
        print("Base multivariate:", self.data.shape, '--->', res.shape)

    def test_k_best(self):
        for i in [5, 10, 20]:
            res = DISRWithMassive(i).run(self.data, self.target)  # Temporary solution
            assert i == res.shape[1]

        for i in [5, 10, 20]:
            f = MultivariateFilter(MIM, i)
            f.fit(self.data, self.target)
            res = f.transform(self.data)
            assert i == res.shape[1]

        for i in [5, 10, 20]:
            f = TraceRatioFisher(i)
            res = self.data[:, f.run(self.data, self.target)[0]]  # Temporary solution
            assert i == res.shape[1]

    def test_measures(self):
        # Multivariate
        for measure in GLOB_MEASURE:
            beta = 0.3 if measure in ['MIFS', 'generalizedCriteria'] else None
            gamma = 0.4 if measure == 'generalizedCriteria' else None
            f = MultivariateFilter(measure, 10, beta, gamma)
            f.fit(self.data, self.target)
            res = f.transform(self.data)
            assert self.data.shape[0] == res.shape[0] and res.shape[1] == 10

    def test_df(self):
        # FCBF
        f = FCBFDiscreteFilter()
        df = f.run(pd.self.dataFrame(self.data), pd.self.dataFrame(self.target))

        f = FCBFDiscreteFilter()
        arr = f.run(self.data, self.target)
        np.testing.assert_array_equal(df, arr)

        # DISR
        f = DISRWithMassive()
        df = f.run(pd.self.dataFrame(self.data), pd.self.dataFrame(self.target))

        f = DISRWithMassive()
        arr = f.run(self.data, self.target)
        np.testing.assert_array_equal(df, arr)

        # Multivariate
        f = MultivariateFilter(MIM, 10)
        f.fit(pd.self.dataFrame(self.data), pd.self.dataFrame(self.target))
        df = f.transform(self.data)

        f = MultivariateFilter(MIM, 10)
        f.fit(pd.self.dataFrame(self.data), pd.self.dataFrame(self.target))
        arr = f.transform(self.data)
        np.testing.assert_array_equal(df, arr)

        # TraceRatio
        f = TraceRatioFisher(10)
        df = f.run(pd.self.dataFrame(self.data), pd.self.dataFrame(self.target))

        f = TraceRatioFisher(10)
        arr = f.run(self.data, self.target)
        np.testing.assert_array_equal(df, arr)


if __name__ == "__main__":
    unittest.main()
