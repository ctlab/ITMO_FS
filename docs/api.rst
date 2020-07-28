######################
ITMO_FS API
######################

This is the full API documentation of the `ITMO_FS` toolbox.

.. _filters_ref:

:mod:`ITMO_FS.filters`: Filter methods
======================================

.. automodule:: filters
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

:mod:`ITMO_FS.filters.univariate`: Univariate filter methods
------------------------------------------------------------

.. automodule:: filters.univariate
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    filters.univariate.VDM
    filters.univariate.UnivariateFilter

Measures for univariate filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: filters.univariate.measures
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS


.. autosummary::
   :toctree: generated/
   :template: function.rst

    filters.univariate.fit_criterion_measure
    filters.univariate.f_ratio_measure
    filters.univariate.gini_index
    filters.univariate.su_measure
    filters.univariate.spearman_corr
    filters.univariate.pearson_corr
    filters.univariate.fechner_corr
    filters.univariate.kendall_corr
    filters.univariate.reliefF_measure
    filters.univariate.chi2_measure
    filters.univariate.information_gain

Cutting rules for univariate filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: filters.univariate.measures
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS


.. autosummary::
   :toctree: generated/
   :template: function.rst

    filters.univariate.select_best_by_value
    filters.univariate.select_worst_by_value
    filters.univariate.select_k_best
    filters.univariate.select_k_worst
    filters.univariate.select_best_percentage
    filters.univariate.select_worst_percentage


:mod:`ITMO_FS.filters.multivariate`: Multivariate filter methods
----------------------------------------------------------------

.. automodule:: filters.multivariate
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    filters.multivariate.DISRWithMassive
    filters.multivariate.FCBFDiscreteFilter
    filters.multivariate.MultivariateFilter
    filters.multivariate.STIR
    filters.multivariate.TraceRatioFisher
    filters.multivariate.MIMAGA


Measures for multivariate filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: filters.multivariate.measures
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS


.. autosummary::
   :toctree: generated/
   :template: function.rst

    filters.multivariate.MIM
    filters.multivariate.MRMR
    filters.multivariate.JMI
    filters.multivariate.CIFE
    filters.multivariate.MIFS
    filters.multivariate.CMIM
    filters.multivariate.ICAP
    filters.multivariate.DCSF
    filters.multivariate.CFR
    filters.multivariate.MRI
    filters.multivariate.IWFS
    filters.multivariate.generalizedCriteria


:mod:`ITMO_FS.filters.unsupervised`: Unsupervised filter methods
----------------------------------------------------------------

.. automodule:: filters.unsupervised
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS


.. autosummary::
   :toctree: generated/
   :template: class.rst

    filters.unsupervised.TraceRatioLaplacian


:mod:`ITMO_FS.filters.sparse`: Sparse filter methods
----------------------------------------------------------------

.. automodule:: filters.sparse
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS


.. autosummary::
   :toctree: generated/
   :template: class.rst

    filters.sparse.MCFS
    filters.sparse.NDFS
    filters.sparse.RFS
    filters.sparse.SPEC
    filters.sparse.UDFS



.. _ensembles_ref:

:mod:`ITMO_FS.ensembles`: Ensemble methods
==========================================

.. automodule:: ensembles
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

:mod:`ITMO_FS.ensembles.measure_based`: Measure based ensemble methods
-----------------------------------------------------------------------

.. automodule:: ensembles.measure_based
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensembles.measure_based.WeightBased


:mod:`ITMO_FS.ensembles.model_based`: Model based ensemble methods
------------------------------------------------------------------

.. automodule:: ensembles.model_based
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensembles.model_based.BestSum


:mod:`ITMO_FS.ensembles.ranking_based`: Ranking based ensemble methods
----------------------------------------------------------------------

.. automodule:: ensembles.ranking_based
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensembles.ranking_based.Mixed


.. _embedded_ref:

:mod:`ITMO_FS.embedded`: Embedded methods
=========================================

.. automodule:: embedded
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    embedded.MOS


.. _hybrid_ref:

:mod:`ITMO_FS.hybrid`: Hybrid methods
=========================================

.. automodule:: hybrid
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    hybrid.FilterWrapperHybrid
    hybrid.Melif


.. _wrappers_ref:

:mod:`ITMO_FS.wrappers`: Wrapper methods
========================================

.. automodule:: wrappers
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

:mod:`ITMO_FS.wrappers.deterministic`: Deterministic wrapper methods
--------------------------------------------------------------------

.. automodule:: wrappers.deterministic
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    wrappers.deterministic.AddDelWrapper
    wrappers.deterministic.BackwardSelection
    wrappers.deterministic.RecursiveElimination
    wrappers.deterministic.SequentialForwardSelection

Deterministic wrapper function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: function.rst

    wrappers.deterministic.qpfs_wrapper



:mod:`ITMO_FS.wrappers.randomized`: Randomized wrapper methods
------------------------------------------------------------------

.. automodule:: wrappers.randomized
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

    wrappers.randomized.HillClimbingWrapper
    wrappers.randomized.SimulatedAnnealing
    wrappers.randomized.TPhMGWO
