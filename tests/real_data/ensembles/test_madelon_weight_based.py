import pytest

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.filters.univariate import UnivariateFilter, fechner_corr, pearson_corr, select_k_best, spearman_corr


pytestmark = pytest.mark.real_data


def test_weight_based_runs_on_madelon_subset(madelon_data):
    x, y = madelon_data
    x = x[:150, :40]
    y = y[:150]

    ensemble = WeightBased(
        [
            UnivariateFilter(fechner_corr),
            UnivariateFilter(spearman_corr),
            UnivariateFilter(pearson_corr),
        ],
        cutting_rule=select_k_best(8),
    ).fit(x, y)

    transformed = ensemble.transform(x)

    assert transformed.shape == (150, 8)
