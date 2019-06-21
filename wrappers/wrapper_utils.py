#!/usr/bin/env python


def get_current_accuracy(__estimator__, X, current_features, test_x, test_y):
    """
    Checks the Accuracy of the Current Features
        Parameters
        ----------
        X : array-like, shape (n_features,n_samples)
            The training input samples.
        current_features : array-like, shape (n_features,n_samples)
            Current Set of Features
        test_x : array-like, shape (n_features, n_samples)
            Testing Set
        test_y : array-like , shape(n_samples)
            Labels
        Returns
        ------
        float, Accuracy of the Current Features

    """

    correct = 0
    for i in range(test_x.length):
        predict = __estimator__.predict([X[j] for j in current_features])
        if predict == test_y[i]:
            correct += 1
    current_accuracy = correct / test_x.length
    return current_accuracy