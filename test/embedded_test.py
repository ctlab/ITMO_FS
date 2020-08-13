import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

from ITMO_FS.embedded import *

np.random.seed(42)

class TestCases(unittest.TestCase):
    data, target = np.random.randint(10, size=(100, 20)), np.random.randint(10, size=(100,))
    feature_names = [''.join(['f', str(i)]) for i in range(data.shape[1])]
    feature_names_override = [''.join(['g', str(i)]) for i in range(data.shape[1])]

    def test_MOSS(self):
        # MOSS
        res = MOS().fit_transform(self.data, self.target, sampling=True)
        assert self.data.shape[0] == res.shape[0]
        print("MOSS:", self.data.shape, '--->', res.shape)

    def test_MOSNS(self):
        # MOSNS
        res = MOS().fit_transform(self.data, self.target, sampling=False)
        assert self.data.shape[0] == res.shape[0]
        print("MOSNS:", self.data.shape, '--->', res.shape)

    def test_losses(self):
        for loss in ['log', 'hinge']:
            res = MOS(loss=loss).fit_transform(self.data, self.target)
            assert self.data.shape[0] == res.shape[0]

    def test_df(self):
        f = MOS()

        df = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target), sampling=True)
        arr = f.fit_transform(self.data, self.target, sampling=True)
        np.testing.assert_array_equal(df, arr)

        df = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target), sampling=False)
        arr = f.fit_transform(self.data, self.target, sampling=False)
        np.testing.assert_array_equal(df, arr)

    def test_pipeline(self):
        # FS
        p = Pipeline([('FS1', MOS())])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0]

        # FS - estim
        p = Pipeline([('FS1', MOS()), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

        # FS - FS
        p = Pipeline([('FS1', MOS(loss='log')), ('FS2', MOS(loss='hinge'))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0]

        # FS - FS - estim
        p = Pipeline([('FS1', MOS(loss='log')), ('FS2', MOS(loss='hinge')), ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

    def test_feature_names_np(self):
        f = MOS()

        arr = f.fit_transform(self.data, self.target, feature_names=self.feature_names, sampling=True)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

        arr = f.fit_transform(self.data, self.target, feature_names=self.feature_names, sampling=False)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df(self):
        f = MOS()

        arr = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target), feature_names=self.feature_names, sampling=True)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

        arr = f.fit_transform(pd.DataFrame(self.data), pd.DataFrame(self.target), feature_names=self.feature_names, sampling=False)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df_defined(self):
        dfX = pd.DataFrame(self.data)
        dfX.columns = self.feature_names
        f = MOS()

        arr = f.fit_transform(dfX, pd.DataFrame(self.target), sampling=True)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

        arr = f.fit_transform(dfX, pd.DataFrame(self.target), sampling=False)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

    def test_feature_names_df_defined_override(self):
        dfX = pd.DataFrame(self.data)
        dfX.columns = self.feature_names_override
        f = MOS()

        arr = f.fit_transform(dfX, pd.DataFrame(self.target), feature_names=self.feature_names, sampling=True)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])

        arr = f.fit_transform(dfX, pd.DataFrame(self.target), feature_names=self.feature_names, sampling=False)
        assert np.all([feature in self.feature_names for feature in f.get_feature_names()])


if __name__ == "__main__":
    unittest.main()
