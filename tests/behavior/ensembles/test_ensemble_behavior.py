import pytest
from sklearn.linear_model import LogisticRegression

from ITMO_FS.ensembles import BestSum, WeightBased
from ITMO_FS.filters.univariate import UnivariateFilter, fechner_corr, pearson_corr, select_k_best, spearman_corr


pytestmark = pytest.mark.behavior


def test_weight_based_prioritizes_informative_features(informative_classification_data):
    x, y = informative_classification_data

    ensemble = WeightBased(
        [
            UnivariateFilter(fechner_corr),
            UnivariateFilter(spearman_corr),
            UnivariateFilter(pearson_corr),
        ],
        cutting_rule=select_k_best(3),
    ).fit(x, y)

    selected = set(ensemble.selected_features_.tolist())
    assert len(selected.intersection({0, 1, 2})) >= 2


def test_best_sum_prioritizes_informative_features(informative_classification_data):
    x, y = informative_classification_data

    ensemble = BestSum(
        [LogisticRegression(max_iter=1000)],
        select_k_best(3),
        lambda model: model.coef_[0] ** 2,
        cv=2,
    ).fit(x, y)

    selected = set(ensemble.selected_features_.tolist())
    assert len(selected.intersection({0, 1, 2})) >= 2
