import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from ITMO_FS.filters.sparse import *

np.random.seed(42)


class TestCases(unittest.TestCase):
    data, target = np.random.randint(10, size=(100, 20)), np.random.randint(10, size=(100,))

    def test_MCFS(self):
        # MCFS
        res = MCFS(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("MCFS:", self.data.shape, '--->', res.shape)

    def test_NDFS(self):
        # NDFS
        res = NDFS(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("NDFS:", self.data.shape, '--->', res.shape)

    def test_RFS(self):
        # RFS
        res = RFS(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("RFS:", self.data.shape, '--->', res.shape)

    def test_SPEC(self):
        # SPEC
        res = SPEC(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("SPEC:", self.data.shape, '--->', res.shape)

    def test_UDFS(self):
        # UDFS
        res = UDFS(10).fit_transform(self.data, self.target)
        assert self.data.shape[0] == res.shape[0]
        print("UDFS:", self.data.shape, '--->', res.shape)

    def test_df(self):
        for f in [MCFS(10), NDFS(10), RFS(10), SPEC(10), UDFS(10)]:
            df = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target))
            arr = f.fit_transform(self.data, self.target)
            np.testing.assert_array_equal(df, arr)

    def test_pipeline(self):
        # FS
        p = Pipeline([('FS1', MCFS(10))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0] and res.shape[1] == 10

        # FS - estim
        p = Pipeline([('FS1', NDFS(10)), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

        # FS - FS
        p = Pipeline([('FS1', RFS(10)), ('FS2', SPEC(5))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0] and res.shape[1] == 5

        # FS - FS - estim
        p = Pipeline([('FS1', UDFS(10)), ('FS2', MCFS(5)), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

    def test_est(self):
        for f in [MCFS(2), NDFS(2), RFS(2), SPEC(2, gamma=np.square), UDFS(2)]:
            check_estimator(f)


if __name__ == "__main__":
    unittest.main()
