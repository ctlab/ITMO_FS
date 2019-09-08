#!/usr/bin/env python

import numpy as np


class FRatioFilter:
    def __init__(self, n=10):
        """
        Insert the value of n i.e the number of features you want to select
        Usage
        >>> frf = FRatioFilter()
        >>> frf.fit(x_data, y_data)
        >>> new_X = frf.tranform(x_data, y_data)
        >>> frf.selection_indexes # TO check the selected indices
        :param n: int
        """
        self.n = n
        self.x_data = []
        self.y_data = []
        self.f_ratios = []
        self.selection_indeces = []

    def __calculate_F_ratio__(self, row, y_data):
        """
        Calculates the Fisher ratio of the row passed to the data
        :param row: ndarray, feature
        :param y_data: ndarray, labels
        :return: int, fisher_ratio
        """
        Mu = np.mean(row)
        inter_class = 0.0
        intra_class = 0.0
        for value in np.unique(y_data):
            index_for_this_value = np.where(y_data == value)[0]
            n = np.sum(row[index_for_this_value])
            mu = np.mean(row[index_for_this_value])
            var = np.var(row[index_for_this_value])
            inter_class += n * np.power((mu - Mu), 2)
            intra_class += (n - 1) * var

        f_ratio = inter_class / intra_class
        return f_ratio

    def fit(self, x_data, y_data):
        """
        Fit the filter to the data
        :param x_data: ndarray, Data with all the features
        :param y_data: ndarray, Label Data
        :return:
        """
        self.x_data, self.y_data = x_data, y_data
        for feature in x_data.T:
            f_ratio = self.__calculate_F_ratio__(feature, y_data.T)
            self.f_ratios.append(f_ratio)
        self.f_ratios = np.array(self.f_ratios)
        # return top n f_ratios
        self.selection_indeces = np.argpartition(self.f_ratios, -self.n)[-self.n:]

    def transform(self, x_data, y_data):
        """
        Transform the data with selected features
        :param x_data:
        :param y_data:
        :return:
        """
        return x_data[:, self.selection_indeces]

    def fit_and_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        return self.transform(x_data, y_data)
