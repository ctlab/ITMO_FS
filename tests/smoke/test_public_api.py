import pytest

from ITMO_FS.embedded import MOS
from ITMO_FS.ensembles import BestSum, Mixed, WeightBased
from ITMO_FS.filters.multivariate import DISRWithMassive, FCBFDiscreteFilter, MultivariateFilter, STIR, TraceRatioFisher
from ITMO_FS.filters.univariate import NDFS, RFS, SPEC, UnivariateFilter, VDM
from ITMO_FS.filters.unsupervised import MCFS, TraceRatioLaplacian, UDFS
from ITMO_FS.hybrid import EGSA, FilterWrapperHybrid, IWSSr_SFLA, Melif
from ITMO_FS.wrappers.deterministic import AddDelWrapper, BackwardSelection, QPFSWrapper, RecursiveElimination, SequentialForwardSelection
from ITMO_FS.wrappers.randomized import HillClimbingWrapper, SimulatedAnnealing, TPhMGWO


pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "exported",
    [
        UnivariateFilter,
        VDM,
        NDFS,
        RFS,
        SPEC,
        MultivariateFilter,
        DISRWithMassive,
        FCBFDiscreteFilter,
        STIR,
        TraceRatioFisher,
        MCFS,
        UDFS,
        TraceRatioLaplacian,
        SequentialForwardSelection,
        BackwardSelection,
        AddDelWrapper,
        RecursiveElimination,
        QPFSWrapper,
        HillClimbingWrapper,
        SimulatedAnnealing,
        TPhMGWO,
        WeightBased,
        BestSum,
        Mixed,
        MOS,
        FilterWrapperHybrid,
        Melif,
        IWSSr_SFLA,
        EGSA,
    ],
)
def test_public_algorithm_exports_are_classes(exported):
    assert isinstance(exported, type)
