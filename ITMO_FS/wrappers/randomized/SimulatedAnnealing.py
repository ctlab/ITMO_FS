import math
import random

import numpy as np

class SimulatedAnnealing(object):

    """
        Performs feature selection using simulated annealing

        Parameters
        ----------
        seed : integer
            Random seed used to initialize np.random.seed()
        iteration_number : integer
            number of iterations of algorithm
        classifier : classifier used for training and testing on provided datasets
            Note that algorithm implementation assumes that classifier has fit, predict methods
            Default algorithm uses sklearn.neighbors.KNeighborsClassifier
        c : integer
            constant c is used to control the rate of feature perturbation
        init_number_of_features : float
            number of features to initialize start features subset,
            Note: by default (5-10) percents of number of features is used
        
        See Also
        --------
        http://www.feat.engineering/simulated-annealing.html

        examples
        --------
        from sklearn.datasets import make_classification
        from sklearn.model_selection import KFold
        from ITMO_FS.wrappers.randomized import SimulatedAnnealing

        x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
        kf = KFold(n_splits=2)
        sa = SimulatedAnnealing()
        for train_index, test_index in kf.split(x):
            sa.run(x[train_index], y[train_index], x[test_index], y[test_index])
            print(sa.selected_features)

        
    """

    def __init__(self, classifier, score=None, seed=1, iteration_number=100, c=1, init_number_of_features=None):
        self.seed = seed
        self.iteration_number = iteration_number
        self.classifier = classifier
        self.score = score
        self.c = c
        self.init_number_of_features = init_number_of_features

    def __acceptance(self, i, prev_score, cur_score):
        return math.exp(-i / self.c * (prev_score - cur_score) / prev_score)

    def fit(self, train_x, train_y, test_x, test_y):
        """
        Runs the Simulated Annealing algorithm on the specified dataset and fits the classifier.
        
        Parameters
        ----------
        train_x : array-like, shape (n_samples, n_features)
            The input training samples.
        train_y : array-like, shape (n_samples)
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
        random.seed(self.seed)
        feature_number = train_x.shape[1]
        if self.init_number_of_features == None:
            percentage = random.randint(5, 11)
            self.init_number_of_features = int(feature_number * percentage / 100)
        feature_subset = np.unique((np.random.randint(0, feature_number, self.init_number_of_features)))
        self.classifier.fit(train_x[:, feature_subset], train_y)
        if self.score==None:
            prev_score = self.classifier.score(test_x[:, feature_subset], test_y)
        else:
            pred_labels = self.classifier.predict(test_x[:, feature_subset])
            prev_score = self.score(pred_labels, test_y)
        for i in range(self.iteration_number):
            operation = random.randint(0, 1)
            percentage = random.randint(1, 5)
            if operation == 1:
                #inc
                include_number = int(feature_number * (percentage / 100))
                not_included_features = np.array([f for f in np.arange(0, feature_number) if f not in feature_subset])
                cur_subset = np.append(feature_subset, np.random.choice(not_included_features, size=include_number, replace=False))
            else:
                #exc
                exclude_number = int(feature_number * (percentage / 100))
                cur_subset = np.delete(feature_subset, np.random.choice(feature_subset, size=exclude_number, replace=False))
            self.classifier.fit(train_x[:, cur_subset], train_y)
            if self.score==None:
                cur_score = self.classifier.score(test_x[:, cur_subset], test_y)
            else:
                pred_labels = self.classifier.predict(test_x[:, cur_subset])
                cur_score = self.score(pred_labels, test_y)
            if cur_score > prev_score:
                feature_subset = cur_subset
                prev_score = cur_score
            else:
                ruv = random.random()
                if ruv > self.__acceptance(i, prev_score, cur_score):
                    feature_subset = cur_subset
                    prev_score = cur_score
        self.selected_features = feature_subset
        

    def predict(self, test_x):
        """
        Predicts labels on test dataset

        Parameters
        ----------
        test_x : array-like, shape (n_samples, n_features)
            The input testing samples.
        
        Return
        ------
        array-like, shape (n_samples,n_selected_features) : array of feature numbers
        
        """
        return classifier.predict(test_x[:, self.selected_features])





