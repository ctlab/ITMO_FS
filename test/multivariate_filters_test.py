import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

from ITMO_FS.filters.multivariate import *

np.random.seed(42)


class TestCases(unittest.TestCase):
    data, target = np.random.randint(10, size=(100, 20)), np.random.randint(10, size=(100,))
    feature_names = [''.join(['f', str(i)]) for i in range(data.shape[1])]
    feature_names_override = [''.join(['g', str(i)]) for i in range(data.shape[1])]

    def test_FCBF(self):
        # FCBF
        res = FCBFDiscreteFilter().fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("Fast Correlation Based filter:", self.data.shape, '--->', res.shape)

    def test_DISR(self):
        # DISR
        res = DISRWithMassive().fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("Double Input Symmetric Relevance:", self.data.shape, '--->', res.shape)

    def test_trace_ratio(self):
        # TraceRatioFisher
        res = TraceRatioFisher(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("TraceRatio:", self.data.shape, '--->', res.shape)

    def test_stir(self):
        # STIR
        res = STIR(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("STatistical Inference Relief:", self.data.shape, '--->', res.shape)

    def test_base_multivariate(self):
        # Multivariate
        f = MultivariateFilter(MIM, 10)
        f.fit(self.data, self.target)
        res = f.transform(self.data)
        assert self.data.shape[0] == res.shape[0]
        print("Base multivariate:", self.data.shape, '--->', res.shape)

    def test_k_best(self):
        for i in [5, 10, 20]:
            res = DISRWithMassive(i).fit_transform(self.data, self.target)
            assert i == res.shape[1]

        for i in [5, 10, 20]:
            f = MultivariateFilter(MIM, i)
            f.fit(self.data, self.target)
            res = f.transform(self.data)
            assert i == res.shape[1]

        for i in [5, 10, 20]:
            res = TraceRatioFisher(i).fit_transform(self.data, self.target)
            assert i == res.shape[1]

        for i in [5, 10, 20]:
            res = STIR(i).fit_transform(self.data, self.target)
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
        for f in [FCBFDiscreteFilter(), DISRWithMassive(), MultivariateFilter(MIM, 10), TraceRatioFisher(10), STIR(10)]:
            df = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target))
            arr = f.fit_transform(self.data, self.target)
            np.testing.assert_array_equal(df, arr)

    def test_pipeline(self):
        # FS
        p = Pipeline([('FS1', MultivariateFilter(MIM, 10))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0] and res.shape[1] == 10

        # FS - estim
        p = Pipeline([('FS1', FCBFDiscreteFilter()), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

        # FS - FS
        p = Pipeline([('FS1', MultivariateFilter(MIM, 10)), ('FS2', STIR(5))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0] and res.shape[1] == 5

        # FS - FS - estim
        p = Pipeline([('FS1', TraceRatioFisher(10)), ('FS2', DISRWithMassive()), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

    def test_feature_names_np(self):
        for f in [FCBFDiscreteFilter(), DISRWithMassive(), MultivariateFilter(MIM, 10), TraceRatioFisher(10), STIR(10)]:
            arr = f.fit_transform(self.data, self.target, feature_names=self.feature_names)
            assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df(self):
        for f in [FCBFDiscreteFilter(), DISRWithMassive(), MultivariateFilter(MIM, 10), TraceRatioFisher(10), STIR(10)]:
            arr = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target), feature_names=self.feature_names)
            assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df_defined(self):
        dfX = pd.DataFrame(self.data)
        dfX.columns = self.feature_names
        for f in [FCBFDiscreteFilter(), DISRWithMassive(), MultivariateFilter(MIM, 10), TraceRatioFisher(10), STIR(10)]:
            arr = f.fit_transform(dfX, pd.DataFrame(self.target))
            assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df_defined_override(self):
        dfX = pd.DataFrame(self.data)
        dfX.columns = self.feature_names_override
        for f in [FCBFDiscreteFilter(), DISRWithMassive(), MultivariateFilter(MIM, 10), TraceRatioFisher(10), STIR(10)]:
            arr = f.fit_transform(dfX, pd.DataFrame(self.target), feature_names=self.feature_names)
            assert np.all([feature in self.feature_names for feature in f.get_feature_names()])


if __name__ == "__main__":
    unittest.main()
