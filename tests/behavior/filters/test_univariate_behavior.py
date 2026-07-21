import numpy as np
import pytest

from ITMO_FS.filters.univariate import UnivariateFilter, anova, chi2_measure, f_ratio_measure, select_k_best


pytestmark = pytest.mark.behavior


@pytest.mark.parametrize("measure", [f_ratio_measure, anova])
def test_supervised_univariate_filters_prioritize_informative_features(
    informative_classification_data, measure
):
    x, y = informative_classification_data

    selector = UnivariateFilter(measure, select_k_best(3)).fit(x, y)
    selected = set(selector.selected_features_.tolist())

    assert len(selected.intersection({0, 1, 2})) >= 2


def test_chi2_prioritizes_informative_features_on_discrete_data(
    informative_discrete_classification_data,
):
    x, y = informative_discrete_classification_data

    selector = UnivariateFilter(chi2_measure, select_k_best(3)).fit(x, y)
    selected = set(selector.selected_features_.tolist())

    assert len(selected.intersection({0, 1, 2})) >= 2


@pytest.mark.parametrize("measure", [f_ratio_measure, anova])
def test_measure_scores_informative_features_higher_than_noise(
    informative_classification_data, measure
):
    x, y = informative_classification_data

    scores = measure(x, y)

    assert np.mean(scores[:3]) > np.mean(scores[3:])
