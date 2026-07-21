import pytest
from sklearn.linear_model import LogisticRegression

from ITMO_FS.filters.univariate import UnivariateFilter, f_ratio_measure, select_k_best
from ITMO_FS.hybrid import FilterWrapperHybrid
from ITMO_FS.wrappers.deterministic import BackwardSelection, SequentialForwardSelection


pytestmark = pytest.mark.behavior


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SequentialForwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
        lambda: BackwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
    ],
)
def test_deterministic_wrappers_keep_informative_features(
    informative_classification_data, factory
):
    x, y = informative_classification_data

    selector = factory().fit(x, y)
    selected = set(selector.selected_features_.tolist())

    assert len(selected.intersection({0, 1, 2})) >= 2


def test_filter_wrapper_hybrid_keeps_signal_from_informative_features(
    informative_classification_data,
):
    x, y = informative_classification_data

    selector = FilterWrapperHybrid(
        UnivariateFilter(f_ratio_measure, select_k_best(5)),
        BackwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
    ).fit(x, y)
    selected = set(selector.selected_features_.tolist())

    assert len(selected.intersection({0, 1, 2})) >= 2
