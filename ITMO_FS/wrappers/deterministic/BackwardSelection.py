#!/usr/bin/env python
import numpy as np

from ...utils import generate_features


class BackwardSelection:
    """
        Backward Selection removes one feature at a time until the number of features to be removed is reached. On each step,
        the best n-1 features out of n are chosen (according to some estimator metric) and the last one is removed.

        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method.
        n_features : int
            Number of features to be removed.
        measure : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable object / function with signature 
            measure(estimator, X, y) which should return only a single value.

        See Also
        --------


        Examples
        --------

        """

    def __init__(self, estimator, n_features, measure):
        # self.__class__ = type(estimator.__class__.__name__, (self.__class__, estimator.__class__), vars(estimator))
        if not hasattr(estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % estimator)
        self.__estimator = estimator
        self.__n_features = n_features
        self.__measure = measure
        self.selected_features = None

    def fit(self, train_X, train_y, val_X, val_y):
        """
            Fits wrapper.

            Parameters
            ----------
            X : array-like, shape (n_samples,n_features)
                The training input samples.
            y : array-like, shape (n_samples,)
                The target values.
            cv : int
                Number of folds in cross-validation.
            Returns
            ------
            None

            See Also
            --------

            Examples
            --------
            >>> from ITMO_FS.wrappers import BackwardSelection
            >>> from sklearn.linear_model import LogisticRegression
            >>> from sklearn.datasets import make_classification
            >>> import numpy as np
            >>> dataset = make_classification(n_samples=100, n_features=20, \
n_informative=4, n_redundant=0, shuffle=False)
            >>> data, target = np.array(dataset[0]), np.array(dataset[1])
            >>> model = BackwardSelection(LogisticRegression(), 15, 'f1_macro')
            >>> model.fit(data, target)
            >>> print(model.selected_features)
            [ 1  2  3 10 15]

        """
        self.selected_features = generate_features(train_X)
        target_size = len(self.selected_features) - self.__n_features

        while len(self.selected_features) != target_size:
            max_measure = 0
            to_delete = 0
            for i in range(len(self.selected_features)):
                iteration_features = np.delete(self.selected_features, i)
                self.__estimator.fit(train_X[:, iteration_features], train_y)
                iteration_measure = self.__measure(self.__estimator.predict(val_X[:, iteration_features]), val_y)
                if iteration_measure > max_measure:
                    max_measure = iteration_measure
                    to_delete = i
            self.selected_features = np.delete(self.selected_features, to_delete)

    def predict(self, X):
        self.__estimator.predict(X[:, self.selected_features])
