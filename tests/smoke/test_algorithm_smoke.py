import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from ITMO_FS.embedded import MOS
from ITMO_FS.ensembles import BestSum, Mixed, WeightBased
from ITMO_FS.filters.multivariate import DISRWithMassive, FCBFDiscreteFilter, MultivariateFilter, STIR, TraceRatioFisher
from ITMO_FS.filters.univariate import (
    NDFS,
    RFS,
    SPEC,
    UnivariateFilter,
    VDM,
    chi2_measure,
    fechner_corr,
    f_ratio_measure,
    gini_index,
    pearson_corr,
    select_k_best,
    spearman_corr,
)
from ITMO_FS.filters.unsupervised import MCFS, TraceRatioLaplacian, UDFS
from ITMO_FS.hybrid import EGSA, FilterWrapperHybrid, IWSSr_SFLA, Melif
from ITMO_FS.wrappers.deterministic import AddDelWrapper, BackwardSelection, QPFSWrapper, RecursiveElimination, SequentialForwardSelection
from ITMO_FS.wrappers.randomized import HillClimbingWrapper, SimulatedAnnealing, TPhMGWO


pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UnivariateFilter(f_ratio_measure, select_k_best(3)),
        lambda: NDFS(n_features=3, max_iterations=5),
        lambda: RFS(n_features=3, max_iterations=5),
        lambda: SPEC(n_features=3),
        lambda: MultivariateFilter("MRMR", 3),
        lambda: STIR(3),
        lambda: TraceRatioFisher(3),
        lambda: MCFS(3, k=2, p=2),
        lambda: UDFS(3, max_iterations=5),
        lambda: TraceRatioLaplacian(3),
        lambda: SequentialForwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
        lambda: BackwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
        lambda: AddDelWrapper(LogisticRegression(max_iter=1000), "accuracy", cv=2, d=1),
        lambda: RecursiveElimination(
            LinearSVC(dual=False),
            3,
            "accuracy",
            lambda model: np.square(model.coef_).sum(axis=0),
            cv=2,
        ),
        lambda: HillClimbingWrapper(LogisticRegression(max_iter=1000), "accuracy", cv=2),
        lambda: SimulatedAnnealing(
            LogisticRegression(max_iter=1000),
            "accuracy",
            cv=2,
            iteration_number=5,
        ),
        lambda: TPhMGWO(
            LogisticRegression(max_iter=1000),
            "accuracy",
            cv=2,
            wolf_number=4,
            iteration_number=2,
        ),
        lambda: WeightBased(
            [
                UnivariateFilter(fechner_corr),
                UnivariateFilter(spearman_corr),
                UnivariateFilter(pearson_corr),
            ],
            cutting_rule=select_k_best(3),
        ),
        lambda: BestSum(
            [LogisticRegression(max_iter=1000)],
            select_k_best(3),
            lambda model: np.square(model.coef_).sum(axis=0),
            cv=2,
        ),
        lambda: MOS(
            SGDClassifier(),
            lambda model: np.square(model.coef_).sum(axis=0),
            epochs=50,
            alphas=np.array([0.01, 0.05]),
            sampling=False,
        ),
        lambda: FilterWrapperHybrid(
            UnivariateFilter(f_ratio_measure, select_k_best(5)),
            BackwardSelection(LogisticRegression(max_iter=1000), 3, "accuracy", cv=2),
        ),
        lambda: EGSA(n_agents=4, iterations=2, stop_iterations=2),
    ],
)
def test_continuous_algorithms_fit_transform_smoke(classification_data, factory):
    x, y = classification_data
    selector = factory().fit(x, y)
    transformed = selector.transform(x)

    assert transformed.shape[0] == x.shape[0]
    assert transformed.shape[1] > 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: VDM(),
        lambda: FCBFDiscreteFilter(delta=0.0),
        lambda: DISRWithMassive(3),
        lambda: Mixed([gini_index, chi2_measure], 3),
        lambda: Melif(
            LogisticRegression(max_iter=1000),
            "accuracy",
            select_k_best(3),
            [
                UnivariateFilter(fechner_corr),
                UnivariateFilter(spearman_corr),
            ],
            delta=0.5,
            cv=2,
        ),
        lambda: IWSSr_SFLA(
            LogisticRegression(max_iter=1000),
            cv=2,
            sfla_m=2,
            sfla_n=4,
            sfla_q=2,
            iterations=1,
            iterations_leaps=1,
        ),
    ],
)
def test_discrete_algorithms_fit_transform_smoke(discrete_classification_data, factory):
    x, y = discrete_classification_data
    selector = factory().fit(x.astype(int), y)
    transformed = selector.transform(x)

    assert transformed.shape[0] == x.shape[0]
    assert transformed.shape[1] > 0


def test_qpfs_wrapper_smoke(classification_data, monkeypatch):
    x, y = classification_data

    def fake_qpfs_body(X, y, fn, alpha=None, r=None, sigma=None, solv=None):
        return np.linspace(1.0, float(X.shape[1]), X.shape[1])

    monkeypatch.setattr(
        "ITMO_FS.wrappers.deterministic.qpfs_wrapper.qpfs_body",
        fake_qpfs_body,
    )

    wrapper = QPFSWrapper(alpha=0.5)
    wrapper.estimator = DummyClassifier(strategy="most_frequent")
    wrapper.fit(x, y)

    transformed = wrapper.transform(x)

    assert transformed.shape == x.shape
    assert wrapper.selected_features_.shape == (x.shape[1],)
