.. -*- mode: rst -*-


.. image:: docs/logos/itmo_small_white_eng.png
  :scale: 10 %
  :target: https://en.itmo.ru/



ITMO_FS
=======

Feature selection library in Python

Package information: |Python 2.7| |Python 3.6| |License| |CircleCI|


Install with

::

   pip install ITMO_FS

Current available algorithms:

+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Filters                              | Wrappers                     | Hybrid          | Embedded | Ensembles       |
+======================================+==============================+=================+==========+=================+
| Spearman correlation                 | Add Del                      | Filter Wrapper  | MOSNS    | MeLiF           |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Pearson correlation                  | Backward selection           |                 | MOSS     | Best goes first |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Fit Criterion                        | Sequential Forward Selection |                 | RFE      | Best sum        |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| F ratio                              | QPFS                         |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Gini index                           | Hill climbing                |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Information Gain                     |                              |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| Minimum Redundancy Maximum Relevance |                              |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| VDM                                  |                              |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+
| QPFS                                 |                              |                 |          |                 |
+--------------------------------------+------------------------------+-----------------+----------+-----------------+

Documentation:

https://itmo-fs.readthedocs.io/en/latest/

.. |Python 2.7| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. |Python 3.6| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. |License| image:: https://img.shields.io/badge/license-BSD%20License-blue.svg
.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn/tree/master
