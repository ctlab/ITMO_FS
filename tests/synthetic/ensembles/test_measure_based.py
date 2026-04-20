import pytest

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.filters.univariate import UnivariateFilter, fechner_corr, pearson_corr, select_k_best, spearman_corr


pytestmark = pytest.mark.synthetic


def test_weight_based_ensemble_smoke(classification_data):
    x, y = classification_data
    filters = [
        UnivariateFilter(fechner_corr),
        UnivariateFilter(spearman_corr),
        UnivariateFilter(pearson_corr),
    ]

    ensemble = WeightBased(filters, cutting_rule=select_k_best(4)).fit(x, y)
    transformed = ensemble.transform(x)

    assert len(ensemble) == 3
    assert ensemble.selected_features_.shape == (4,)
    assert transformed.shape == (x.shape[0], 4)
