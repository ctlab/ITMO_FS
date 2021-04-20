import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from ITMO_FS.embedded import *
from ITMO_FS.utils import weight_func

np.random.seed(42)


class TestCases(unittest.TestCase):
    data, target = np.random.randint(
        10, size=(
            100, 20)), np.random.randint(
        10, size=(
            100,))

    def test_MOS_err_loss(self):
        with self.assertRaises(KeyError):
            MOS(model=SGDClassifier(), weight_func=weight_func,
                sampling=True, loss="err").fit(self.data,
                                               self.target)

    def test_MOS_no_sampling(self):
        # MOSS
        res = MOS(
            model=SGDClassifier(),
            weight_func=weight_func).fit_transform(
            self.data,
            self.target)
        assert self.data.shape[0] == res.shape[0]
        print("MOSS:", self.data.shape, '--->', res.shape)

    def test_MOSS(self):
        # MOSS
        res = MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=True).fit_transform(
            self.data,
            self.target)
        assert self.data.shape[0] == res.shape[0]
        print("MOSS:", self.data.shape, '--->', res.shape)

    def test_MOSS_n_naigbours_err(self):
        # MOSS
        with self.assertRaises(ValueError):
            MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=True,k_neighbors=1000).fit_transform(
            self.data,
            self.target)


    def test_MOSS_hinge(self):
        # MOSS
        res = MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=True,loss="hinge").fit_transform(
            self.data,
            self.target)
        assert self.data.shape[0] == res.shape[0]
        print("MOSS:", self.data.shape, '--->', res.shape)

    def test_MOSNS(self):
        # MOSNS
        res = MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=False).fit_transform(
            self.data,
            self.target)
        assert self.data.shape[0] == res.shape[0]
        print("MOSNS:", self.data.shape, '--->', res.shape)

    def test_losses(self):
        for loss in ['log', 'hinge']:
            res = MOS(
                model=SGDClassifier(),
                weight_func=weight_func,
                loss=loss).fit_transform(
                self.data,
                self.target)
            assert self.data.shape[0] == res.shape[0]

    def test_df(self):
        f = MOS(model=SGDClassifier(), weight_func=weight_func, sampling=True)

        df = f.fit_transform(
            pd.DataFrame(
                self.data), pd.DataFrame(
                self.target))
        arr = f.fit_transform(self.data, self.target)
        np.testing.assert_array_equal(df, arr)

        f = MOS(model=SGDClassifier(), weight_func=weight_func, sampling=False)

        df = f.fit_transform(
            pd.DataFrame(
                self.data), pd.DataFrame(
                self.target))
        arr = f.fit_transform(self.data, self.target)
        np.testing.assert_array_equal(df, arr)

    def test_pipeline(self):
        # FS
        p = Pipeline(
            [('FS1', MOS(model=SGDClassifier(), weight_func=weight_func))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0]

        # FS - estim
        p = Pipeline([('FS1', MOS(model=SGDClassifier(),
                                  weight_func=weight_func)),
                      ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

        # FS - FS
        p = Pipeline([('FS1',
                       MOS(model=SGDClassifier(),
                           weight_func=weight_func,
                           loss='log')),
                      ('FS2',
                       MOS(model=SGDClassifier(),
                           weight_func=weight_func,
                           loss='hinge'))])
        p.fit(self.data, self.target)
        res = p.transform(self.data)
        assert self.data.shape[0] == res.shape[0]

        # FS - FS - estim
        p = Pipeline([('FS1',
                       MOS(model=SGDClassifier(), weight_func=weight_func,
                           loss='log')), ('FS2', MOS(
            model=SGDClassifier(), weight_func=weight_func, loss='hinge')),
                      ('E1', LogisticRegression())])
        p.fit(self.data, self.target)
        assert 0 <= p.score(self.data, self.target) <= 1

    def test_est(self):
        moss = MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=True)
        mosns = MOS(
            model=SGDClassifier(),
            weight_func=weight_func,
            sampling=False)

        # for some reason using local weight_func or lambda here causes it to fail with pickle errors
        # so we're using an imported weight_func
        check_estimator(moss)
        check_estimator(mosns)


if __name__ == "__main__":
    unittest.main()
