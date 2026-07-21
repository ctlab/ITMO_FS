import pytest
from sklearn.linear_model import LogisticRegression

from ITMO_FS.filters.univariate import UnivariateFilter, fechner_corr, select_k_best, spearman_corr
from ITMO_FS.hybrid import Melif


pytestmark = pytest.mark.behavior


def test_melif_keeps_informative_features(informative_discrete_classification_data):
    x, y = informative_discrete_classification_data

    selector = Melif(
        LogisticRegression(max_iter=1000),
        "accuracy",
        select_k_best(3),
        [
            UnivariateFilter(fechner_corr),
            UnivariateFilter(spearman_corr),
        ],
        delta=0.5,
        cv=2,
    ).fit(x, y)

    selected = set(selector.selected_features_.tolist())
    assert len(selected.intersection({0, 1, 2})) >= 2
