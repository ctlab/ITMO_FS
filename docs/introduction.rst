.. _introduction:

============
Introduction
============

.. _api_ITMO_FS:

API's of feature selectors
----------------------------------

Available selectors follow the scikit-learn API using the base estimator
and selector mixin:

:Transformer:

    The base object, implements a ``fit`` method to learn from data, either::

      selector.fit(data, targets)

    To select features from a data set after learning, each selector implements::

      data_selected = selector.transform(data)

    To learn from data and select features from the same data set at once, each selector implements::

      data_selected = selector.fir_transform(data, targets)

    To reverse the selection operation, each selector implements::

      data_reversed = selector.fir_transform(data)

Feature selectors accept the same inputs that in scikit-learn:

* ``data``: array-like (2-D list, pandas.Dataframe, numpy.array) or sparse
  matrices;
* ``targets``: array-like (1-D list, pandas.Series, numpy.array).

The output will be of the following type:

* ``data_selected``: array-like (2-D list, pandas.Dataframe, numpy.array) or
   sparse matrices;
* ``data_reversed``: array-like (2-D list, pandas.Dataframe, numpy.array) or
   sparse matrices;

.. topic:: Sparse input

   For sparse input the data is **converted to the Compressed Sparse Rows
   representation** (see ``scipy.sparse.csr_matrix``) before being fed to the
   sampler. To avoid unnecessary memory copies, it is recommended to choose the
   CSR representation upstream.

.. _problem_statement:

Problem statement regarding data sets with redundant features
------------------------------------------------

Feature selection methods can be used to identify and remove unneeded,
irrelevant and redundant attributes from data that do not contribute
to the accuracy of a predictive model or may in fact decrease the
accuracy of the model. Fewer attributes is desirable because it reduces
the complexity of the model, and a simpler model is simpler to understand
and explain.

Here is one of examples of feature selection improving the classification quality::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import SGDClassifier
    >>> from ITMO_FS.embedded import MOS

    >>> X, y = make_classification(n_samples=300, n_features=10, random_state=0, n_informative=2)
    >>> sel = MOS()
    >>> trX = sel.fit_transform(X, y, smote=False)

    >>> cl1 = SGDClassifier()
    >>> cl1.fit(X, y)
    >>> cl1.score(X, y)
    0.9033333333333333

    >>> cl2 = SGDClassifier()
    >>> cl2.fit(trX, y)
    >>> cl2.score(trX, y)
    0.9433333333333334

As expected, the quality of the SVGClassifier's results is impacted by the presence of redundant features in data set.
We can see that after using of feature selection the mean accuracy increases from 0.903 to 0.943.
