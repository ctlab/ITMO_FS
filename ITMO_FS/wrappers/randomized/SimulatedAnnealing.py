import math

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from ...utils import BaseWrapper


class SimulatedAnnealing(BaseWrapper):
    """
        Performs feature selection using simulated annealing

        Parameters
        ----------
        seed : int
            Random seed used to initialize np.random.seed()
        iteration_number : int
            number of iterations of algorithm
        estimator : Classifier instance
            ``Classifier`` used for training and testing on provided datasets.

            - Note that algorithm implementation assumes that estimator has fit, predict methods. Default algorithm \
            uses ``sklearn.neighbors.KNeighborsClassifier``

        c : int
            constant c is used t o control the rate of feature perturbation
        init_number_of_features : int
            number of features to initialize start features subset,
            Note: by default (5-10) percents of number of features is used
        test_size : float
            The test set size relative to the whole passed dataset.
        
        Notes
        -----
        For more details see `this paper <http://www.feat.engineering/simulated-annealing.html/>`_.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import KFold
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.metrics import f1_score
        >>> from ITMO_FS.wrappers.randomized import SimulatedAnnealing
        >>> x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
        >>> kf = KFold(n_splits=2)
        >>> sa = SimulatedAnnealing()
        >>> for train_index, test_index in kf.split(x):
        ...    sa.fit()
        ...    print(sa.selected_features)
    """

    def __init__(self, estimator, scorer, seed=1, iteration_number=100, c=1, init_number_of_features=None, test_size=0.25):
        self.estimator = estimator
        self.scorer = scorer
        self.seed = seed
        self.iteration_number = iteration_number
        self.c = c
        self.init_number_of_features = init_number_of_features
        self.test_size = test_size

    def __acceptance(self, i, prev_score, cur_score):
        return math.exp(-i / self.c * (prev_score - cur_score) / prev_score)

    def _fit(self, X, y):
        """
            Runs the Simulated Annealing algorithm on the specified dataset and fits the classifier.
            
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input training samples.
            y : array-like, shape (n_samples)
                The classes for training samples.
            test_x : array-like, shape (n_samples, n_features)
                The input testing samples.
            test_y : array-like, shape (n_samples)
                The classes for testing samples.

            Return
            ------
            None
        """
        np.random.seed(self.seed)
        if not hasattr(self.estimator, 'fit'):
            raise TypeError("estimator should be an estimator implementing "
                            "'fit' method, %r was passed" % self.estimator)
        self._estimator = clone(self.estimator)

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=self.test_size)
        feature_number = train_x.shape[1]
        if self.init_number_of_features is None:
            percentage = np.random.randint(5, 11)
            init_number_of_features = int(feature_number * percentage / 100) + 1
        else:
            init_number_of_features = self.init_number_of_features
        feature_subset = np.unique((np.random.randint(0, feature_number, init_number_of_features)))
        prev_score = self.__get_score(train_x, train_y, test_x, test_y, feature_subset)
        for i in range(self.iteration_number):
            operation = np.random.randint(0, 1)
            percentage = np.random.randint(1, 5)
            if operation == 1:
                # inc
                include_number = int(feature_number * (percentage / 100))
                not_included_features = np.array([f for f in np.arange(0, feature_number) if f not in feature_subset])
                cur_subset = np.append(feature_subset,
                                       np.random.choice(not_included_features, size=include_number, replace=False))
            else:
                # exc
                exclude_number = int(feature_number * (percentage / 100))
                cur_subset = np.delete(feature_subset,
                                       np.random.choice(feature_subset, size=exclude_number, replace=False))
            cur_score = self.__get_score(train_x, train_y, test_x, test_y, feature_subset)
            if cur_score > prev_score:
                feature_subset = cur_subset
                prev_score = cur_score
            else:
                ruv = np.random.random()
                if ruv > self.__acceptance(i, prev_score, cur_score):
                    feature_subset = cur_subset
                    prev_score = cur_score
        self.selected_features_ = feature_subset
        self._estimator.fit()

    def __get_score(self, train_x, train_y, test_x, test_y, subset):
        self._estimator.fit()
        pred_labels = self._estimator.predict(test_x[:, subset])
        score = self.scorer(pred_labels, test_y)
        return score
