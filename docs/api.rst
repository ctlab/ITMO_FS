######################
ITMO_FS API
######################

This is the full API documentation of the `ITMO_FS` toolbox.

.. _under_sampling_ref:

:mod:`ITMO_FS.filters`: Feature selection filters
======================================================

.. automodule:: ITMO_FS.filters
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

Filters
--------------------

.. automodule:: ITMO_FS.filters
   :no-members:
   :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

   filters

mod:`ITMO_FS.Univariate`: Univariate filters
====================================================

.. automodule:: ITMO_FS.filters.univariate
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

   over_sampling.ADASYN
   over_sampling.BorderlineSMOTE
   over_sampling.KMeansSMOTE
   over_sampling.RandomOverSampler
   over_sampling.SMOTE
   over_sampling.SMOTENC
   over_sampling.SVMSMOTE


.. _combine_ref:

mod:`ITMO_FS.Multivariate`: Multivariate filters
====================================================

.. automodule:: ITMO_FS.filters.multivariate
    :no-members:
    :no-inherited-members:

.. currentmodule:: ITMO_FS

.. autosummary::
   :toctree: generated/
   :template: class.rst

   filters.multivariate.ADASYN
   filters.multivariate.BorderlineSMOTE
   filters.multivariate.KMeansSMOTE
   filters.multivariate.RandomOverSampler
   filters.multivariate.SMOTE
   filters.multivariate.SMOTENC
   filters.multivariate.SVMSMOTE


.. _combine_ref:

:mod:`ITMO_FS.wrappers`: Feature selection wrappers
======================================================

.. automodule:: imblearn.under_sampling._prototype_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   under_sampling.CondensedNearestNeighbour
   under_sampling.EditedNearestNeighbours
   under_sampling.RepeatedEditedNearestNeighbours
   under_sampling.AllKNN
   under_sampling.InstanceHardnessThreshold
   under_sampling.NearMiss
   under_sampling.NeighbourhoodCleaningRule
   under_sampling.OneSidedSelection
   under_sampling.RandomUnderSampler
   under_sampling.TomekLinks

.. _over_sampling_ref:

: