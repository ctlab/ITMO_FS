import numpy as np
import pytest

from ITMO_FS.filters.univariate import UnivariateFilter, f_ratio_measure, select_k_best


pytestmark = pytest.mark.real_data


def test_univariate_filter_runs_on_madelon_subset(madelon_data):
    x, y = madelon_data
    x = x[:200, :50]
    y = y[:200]

    filt = UnivariateFilter(f_ratio_measure, select_k_best(10)).fit(x, y)
    transformed = filt.transform(x)

    assert transformed.shape == (200, 10)
    assert np.isfinite(filt.feature_scores_).all()
