#!/usr/bin/env python

from numpy import mean, vstack
from numpy.random import shuffle


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
        :param X:
        :param __estimator__:

    """

    correct = 0
    for i in range(test_x.length):
        predict = __estimator__.predict([test_x[j] for j in current_features]) # TODO: rewrite to correct matrix of shape (n_samples,n_features) 
        if predict == test_y[i]:
            correct += 1
    current_accuracy = correct / test_x.length
    return current_accuracy


def get_current_cv_accuracy(__estimator__, X, y, current_features, cv=3):
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

    accuracies = []
    for data, target in cross_validate(X, y, k=cv):
        predict = __estimator__.predict(data[:, current_features]).reshape(target.shape) # TODO: rewrite to correct matrix of shape (n_samples,n_features) or change docs
        correct = sum(predict == target)
        accuracies.append(correct / len(data))

    return mean(accuracies)


def cross_validate(X, y, random=False, k=3):
    x_t, y_t = X, y
    if random:
        shuffle(x_t)
        shuffle(y_t)
    x_y_pairs = []
    n = int(X.shape[0] / k)
    for i in range(k):
        x_y_pairs.append([x_t[i * n:(i + 1) * n], y_t[i * n:(i + 1) * n]])
    for i in range(X.shape[0] % k):
        x_y_pairs[i][0] = vstack((x_y_pairs[i][0], x_t[n * k + i]))
        x_y_pairs[i][1] = vstack((x_y_pairs[i][1], y_t[n * k + i]))
    return x_y_pairs
