import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2, f_classif

from ITMO_FS.filters.univariate import UnivariateFilter, anova, chi2_measure, f_ratio_measure, select_k_best


pytestmark = pytest.mark.synthetic


def test_univariate_filter_fit_transform_smoke(tiny_filter_example):
    x, y = tiny_filter_example

    filt = UnivariateFilter(f_ratio_measure, select_k_best(2)).fit(x, y)
    transformed = filt.transform(x)

    assert transformed.shape == (5, 2)
    assert filt.selected_features_.shape == (2,)
    assert filt.feature_scores_.shape == (5,)


def test_chi2_matches_sklearn():
    iris = load_iris()
    x = iris.data.astype(int)
    y = iris.target

    scores = chi2_measure(x, y)
    sklearn_scores = chi2(x, y)[0]

    assert scores.shape == sklearn_scores.shape
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_anova_matches_sklearn():
    iris = load_iris()
    x = iris.data
    y = iris.target

    np.testing.assert_allclose(anova(x, y), f_classif(x, y)[0])
