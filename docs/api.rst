######################
ITMO_FS API
######################

This is the API reference for the public modules currently shipped in
`ITMO_FS`.

.. currentmodule:: ITMO_FS

.. _filters_ref:

:mod:`ITMO_FS.filters`
======================

Univariate filters
------------------

.. automodule:: ITMO_FS.filters.univariate
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.filters.univariate.UnivariateFilter
    ITMO_FS.filters.univariate.VDM
    ITMO_FS.filters.univariate.NDFS
    ITMO_FS.filters.univariate.RFS
    ITMO_FS.filters.univariate.SPEC

.. autosummary::
   :toctree: generated/
   :template: function.rst

    ITMO_FS.filters.univariate.fit_criterion_measure
    ITMO_FS.filters.univariate.f_ratio_measure
    ITMO_FS.filters.univariate.gini_index
    ITMO_FS.filters.univariate.su_measure
    ITMO_FS.filters.univariate.modified_t_score
    ITMO_FS.filters.univariate.fechner_corr
    ITMO_FS.filters.univariate.information_gain
    ITMO_FS.filters.univariate.relief_measure
    ITMO_FS.filters.univariate.reliefF_measure
    ITMO_FS.filters.univariate.chi2_measure
    ITMO_FS.filters.univariate.spearman_corr
    ITMO_FS.filters.univariate.pearson_corr
    ITMO_FS.filters.univariate.laplacian_score
    ITMO_FS.filters.univariate.kendall_corr
    ITMO_FS.filters.univariate.select_k_best
    ITMO_FS.filters.univariate.select_k_worst
    ITMO_FS.filters.univariate.select_best_by_value
    ITMO_FS.filters.univariate.select_worst_by_value
    ITMO_FS.filters.univariate.select_best_percentage
    ITMO_FS.filters.univariate.select_worst_percentage

Multivariate filters
--------------------

.. automodule:: ITMO_FS.filters.multivariate
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.filters.multivariate.DISRWithMassive
    ITMO_FS.filters.multivariate.FCBFDiscreteFilter
    ITMO_FS.filters.multivariate.MultivariateFilter
    ITMO_FS.filters.multivariate.STIR
    ITMO_FS.filters.multivariate.TraceRatioFisher

Unsupervised filters
--------------------

.. automodule:: ITMO_FS.filters.unsupervised
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.filters.unsupervised.MCFS
    ITMO_FS.filters.unsupervised.TraceRatioLaplacian
    ITMO_FS.filters.unsupervised.UDFS

.. _ensembles_ref:

:mod:`ITMO_FS.ensembles`
========================

.. automodule:: ITMO_FS.ensembles
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.ensembles.WeightBased
    ITMO_FS.ensembles.BestSum
    ITMO_FS.ensembles.Mixed

.. _embedded_ref:

:mod:`ITMO_FS.embedded`
=======================

.. automodule:: ITMO_FS.embedded
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.embedded.MOS

.. _hybrid_ref:

:mod:`ITMO_FS.hybrid`
=====================

.. automodule:: ITMO_FS.hybrid
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.hybrid.FilterWrapperHybrid
    ITMO_FS.hybrid.IWSSr_SFLA
    ITMO_FS.hybrid.Melif

.. _wrappers_ref:

:mod:`ITMO_FS.wrappers`
=======================

Deterministic wrappers
----------------------

.. automodule:: ITMO_FS.wrappers.deterministic
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.wrappers.deterministic.AddDelWrapper
    ITMO_FS.wrappers.deterministic.BackwardSelection
    ITMO_FS.wrappers.deterministic.QPFSWrapper
    ITMO_FS.wrappers.deterministic.RecursiveElimination
    ITMO_FS.wrappers.deterministic.SequentialForwardSelection

Randomized wrappers
-------------------

.. automodule:: ITMO_FS.wrappers.randomized
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ITMO_FS.wrappers.randomized.HillClimbingWrapper
    ITMO_FS.wrappers.randomized.SimulatedAnnealing
    ITMO_FS.wrappers.randomized.TPhMGWO
