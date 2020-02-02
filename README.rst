ITMO_FS
=======

Feature selection library in Python

Package information: |Python 2.7| |Python 3.6| |License|

Install with

::

   pip install ITMO_FS

Current available algorithms:

+--------------------------------------+------------------------------+--------+----------+
| Filters                              | Wrappers                     | Hybrid | Embedded |
+======================================+==============================+========+==========+
| Spearman correlation                 | Add Del                      | MeLiF  | MOSNS    |
+--------------------------------------+------------------------------+--------+----------+
| Pearson correlation                  | Backward selection           |        | MOSS     |
+--------------------------------------+------------------------------+--------+----------+
| Fit Criterion                        | Sequential Forward Selection |        |          |
+--------------------------------------+------------------------------+--------+----------+
| F ratio                              |                              |        |          |
+--------------------------------------+------------------------------+--------+----------+
| Gini index                           |                              |        |          |
+--------------------------------------+------------------------------+--------+----------+
| Information Gain                     |                              |        |          |
+--------------------------------------+------------------------------+--------+----------+
| Minimum Redundancy Maximum Relevance |                              |        |          |
+--------------------------------------+------------------------------+--------+----------+
| VDM                                  |                              |        |          |
+--------------------------------------+------------------------------+--------+----------+


To use basic filter:

::

   from sklearn.datasets import load_iris
   from ITMO_FS.filters import UnivariateFilter, spearman_corr, select_best_by_value # provides you a filter class, basic measures and cutting rules

   data, target = load_iris(True)
   res = UnivariateFilter(spearman_corr, select_best_by_value(0.9999)).run(data, target)
   print("SpearmanCorr:", data.shape, '--->', res.shape)

.. |Python 2.7| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. |Python 3.6| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. |License| image:: https://img.shields.io/badge/license-MIT%20License-blue.svg

