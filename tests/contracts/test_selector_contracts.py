from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

import pytest

from ITMO_FS.ensembles import WeightBased
from ITMO_FS.filters.univariate import UnivariateFilter, fechner_corr, f_ratio_measure, pearson_corr, select_k_best
from ITMO_FS.wrappers.deterministic import SequentialForwardSelection


pytestmark = pytest.mark.contracts


@pytest.mark.parametrize(
    "factory, uses_discrete",
    [
        (lambda: UnivariateFilter(f_ratio_measure, select_k_best(3)), False),
        (
            lambda: WeightBased(
                [
                    UnivariateFilter(fechner_corr),
                    UnivariateFilter(pearson_corr),
                ],
                cutting_rule=select_k_best(3),
            ),
            False,
        ),
        (
            lambda: SequentialForwardSelection(
                LogisticRegression(max_iter=1000),
                n_features=3,
                measure="accuracy",
                cv=2,
            ),
            False,
        ),
    ],
)
def test_transform_before_fit_raises(factory, uses_discrete, classification_data, discrete_classification_data):
    x, _ = discrete_classification_data if uses_discrete else classification_data

    with pytest.raises(NotFittedError):
        factory().transform(x)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UnivariateFilter(f_ratio_measure, select_k_best(3)),
        lambda: SequentialForwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
    ],
)
def test_fit_transform_matches_fit_then_transform(factory, classification_data):
    x, y = classification_data
    estimator = factory()

    transformed_direct = estimator.fit_transform(x, y)
    transformed_separate = clone(factory()).fit(x, y).transform(x)

    assert transformed_direct.shape == transformed_separate.shape
    assert (transformed_direct == transformed_separate).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UnivariateFilter(f_ratio_measure, select_k_best(3)),
        lambda: WeightBased([UnivariateFilter(fechner_corr)], cutting_rule=select_k_best(3)),
        lambda: SequentialForwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
    ],
)
def test_estimators_are_cloneable(factory):
    estimator = factory()
    cloned = clone(estimator)
    assert isinstance(cloned, type(estimator))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UnivariateFilter(f_ratio_measure, select_k_best(3)),
        lambda: WeightBased([UnivariateFilter(fechner_corr)], cutting_rule=select_k_best(3)),
        lambda: SequentialForwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
    ],
)
def test_selected_features_are_unique_and_in_bounds(factory, classification_data):
    x, y = classification_data
    estimator = factory().fit(x, y)
    selected = estimator.selected_features_

    assert selected.ndim == 1
    assert len(set(selected.tolist())) == selected.size
    assert (selected >= 0).all()
    assert (selected < x.shape[1]).all()
