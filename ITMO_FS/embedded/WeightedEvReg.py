import math
import numpy as np
import random
from collections import defaultdict

from ITMO_FS.utils import apply_cr

from ..utils import BaseTransformer


class WeightedEvReg(BaseTransformer):
    """
        Builds weighted evidential regression model, which learns features weights during fitting.
        Thus learnt feature wieghts can be used as ranks in feature selection.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The input samples.
        y : numpy array, shape (n_samples, )
            The classes for the samples.
        alpha : np.float64
            Learning rate (Optional 0.01 by default)
        num_epochs : int
            Number of epochs of gradient descent (Optional 1000 by default)
        p : int
            Power of minkoswki distance (Optional 2 by default)
        k : int
            Number of neighbors for knn-approach optimization (Optional 0.1 from X.shape[0] by default)
        radius : np.float64
            Radius of the RBF distance

        Returns
        -------
        Score for each feature as a numpy array, shape (n_features, )

        See Also
        --------
        https://www.researchgate.net/publication/343493691_Feature_Selection_for_Health_Care_Costs_Prediction_Using_Weighted_Evidential_Regression

        Note:
        The main idea is to use the weighted EVREG for predicting labels and then optimize the weights according to loss via
        gradient descent for fixed number of epochs. The weights are used in counting distance between objects, thus
        weighting features impact in distance values. While optimizing features impact in distance algorithm optimizes
        quality of prediction thus finding the bond between feature and prediction and performing feature selection

        Examples
        --------
        >>> import sklearn.datasets as datasets
        >>> from ITMO_FS.embedded import WeightedEvReg
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> weighted_ev_reg = WeightedEvReg(cutting_rule=('K best', 2), num_epochs=100)
        >>> weighted_ev_reg.fit(X, y)
        >>> print(weighted_ev_reg.selected_features_)
    """

    def __init__(self, cutting_rule, alpha=0.01, num_epochs=1000, p=2, k=None, radius=5.0):
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.p = p
        self.k = k
        self.radius = radius
        self.cutting_rule = cutting_rule
        random.seed(42)

    @staticmethod
    def __weighted_minkowski_distance(first, second, weights, p):
        return sum(abs((first - second) * weights) ** p) ** (1.0 / p)

    @staticmethod
    def __rbf(distance, radius):
        return math.exp(-(distance ** 2) / radius)

    def __rbf_vectors(self, first, second, weights, p, radius):
        return math.exp(-(self.__weighted_minkowski_distance(first, second, weights, p) ** 2) / radius)

    def __count_K(self, X, index, nearest_neighbors, weights, p, radius):
        all_distances = [self.__rbf(self.__weighted_minkowski_distance(X[index], X[t], weights, p), radius) for t in
                         nearest_neighbors]
        distances_minus = np.prod([1 - dist for dist in all_distances])
        distances_without = [dist * distances_minus / (1 - dist) for dist in all_distances]
        return distances_minus + sum(distances_without), distances_without, distances_minus

    @staticmethod
    def __elements_number(k_smallest):
        sum(map(lambda t: len(t), k_smallest.values()))

    def __evreg_predict(self, X, y, index, cur_weights, p, k, radius):
        to_predict = X[index]
        k_smallest = defaultdict(list)
        for i in range(X.shape[0]):
            if i == index:
                continue
            cur_distance = self.__weighted_minkowski_distance(to_predict, X[i], cur_weights, p)
            if self.__elements_number(k_smallest) == k:
                max_smallest = max(k_smallest.keys())
                if cur_distance < max_smallest:
                    del k_smallest[max_smallest][random.randint(0, len(k_smallest[max_smallest]) - 1)]
                    k_smallest[cur_distance] = i
            else:
                k_smallest[cur_distance].append(i)
        nearest_neighbors = list([item for sublist in k_smallest.values() for item in sublist])
        K, distances_without, m_star = self.__count_K(X, index, nearest_neighbors, cur_weights, p, radius)
        m = 1.0 / K * np.array(distances_without)
        return sum(m[i] * y[nearest_neighbors[i]] for i in range(k)) + m_star * (
                max(y[nearest_neighbors]) + min(y[nearest_neighbors])) / 2

    @staticmethod
    def __count_loss(expected_y, predicted_y):
        return 1.0 / len(expected_y) * sum((expected_y - predicted_y) ** 2)

    @staticmethod
    def __minkowski_derivative(first, second, weights, p):
        return sum(abs((first - second) * weights) ** p) ** (1.0 / p - 1) * p / (p - 1) * ((first - second) ** (p - 1))

    def __rbf_derivative(self, first, second, weights, p, radius):
        distance = self.__weighted_minkowski_distance(first, second, weights, p)
        return -2.0 / radius * self.__rbf(distance, radius) * distance * self.__minkowski_derivative(first, second,
                                                                                                     weights, p)

    def __prod_seq_func(self, X, index, skip, weights, p, radius, also_skip=None):
        return np.prod(
            [1 - self.__rbf_vectors(X[index], X[i], weights, p, radius) for i in range(X.shape[0]) if
             i not in skip and i != also_skip])

    def __product_sequence_derivative(self, X, index, skip, weights, p, radius):
        return np.sum(
            [self.__rbf_derivative(index, i, weights, p, radius) * self.__prod_seq_func(X, index, skip, weights, p,
                                                                                        radius, i) for
             i
             in range(X.shape[0]) if
             i not in skip],
            axis=0)

    def __K_derivative(self, X, index, weights, p, radius):
        sum_func = lambda skip: self.__rbf_derivative(X[index], X[skip], weights, p, radius) * \
                                self.__prod_seq_func(X, index, [skip, index], weights, p, radius) + \
                                self.__rbf_vectors(X[index], X[skip], weights, p, radius) * \
                                self.__product_sequence_derivative(X, index, [index, skip], weights, p, radius)

        return self.__product_sequence_derivative(X, index, [index], weights, p, radius) + np.sum(
            [sum_func(i) for i in range(X.shape[0]) if i != index], axis=0)

    def __count_K_all(self, X, index, weights, p, radius):
        all_distances = [self.__rbf(self.__weighted_minkowski_distance(X[index], X[t], weights, p), radius) for t in
                         range(X.shape[0]) if t != index]
        distances_minus = np.prod([1 - dist for dist in all_distances])
        distances_without = [dist * distances_minus / (1 - dist) for dist in all_distances]
        return distances_minus + sum(distances_without), distances_without, distances_minus

    def __single_mass_derivative(self, X, i, j, weights, p, radius):
        K, _, distances_minus = self.__count_K_all(X, i, weights, p, radius)
        return (K * self.__rbf_derivative(X[i], X[j], weights, p, radius) - self.__K_derivative(X, i, weights, p,
                                                                                                radius) *
                self.__rbf_vectors(X[i], X[j], weights, p, radius)) * distances_minus / (K ** 2) + \
               self.__rbf_vectors(X[i], X[j], weights, p, radius) / \
               K * self.__product_sequence_derivative(X, i, [i], weights, p, radius)

    def __mass_star_derivative(self, X, i, weights, p, radius):
        K, _, distances_minus = self.__count_K_all(X, i, weights, p, radius)
        return (K * self.__product_sequence_derivative(X, i, [i], weights, p, radius) -
                self.__K_derivative(X, i, weights, p, radius) * distances_minus) / (K ** 2)

    def __y_derivative(self, X, i, weights, p, radius, y):
        y_der = [0 for i in range(len(weights))]
        y_lab = [y[j] for j in range(len(y)) if j != i]
        for j in range(X.shape[0]):
            if j == i:
                continue
            y_der += self.__single_mass_derivative(X, i, j, weights, p, radius) * y[j]
        y_der += self.__mass_star_derivative(X, i, weights, p, radius) * (max(y_lab) + min(y_lab)) / 2
        return y_der

    def __update_weights(self, X, y, alpha, weights, p, radius):
        return weights + alpha * 2.0 / X.shape[0] * -1 * \
               np.sum([self.__y_derivative(X, i, weights, p, radius, y) for i in range(X.shape[0])], axis=0)

    def _fit(self, X, y):
        """
            Runs the Weighted evidential regression algorithm on the specified dataset.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            y : array-like, shape (n_samples)
                The classes for the samples.

            Returns
            ------
            None
        """
        if self.k is None:
            self.k = int(0.1 * X.shape[0])
            if self.k < 1:
                self.k = X.shape[0] - 1
        print(self.k)
        feature_size = X.shape[1]
        best_weights = np.ones(feature_size, dtype=np.float64)
        min_loss = float('inf')
        cur_weights = best_weights.copy()
        for _ in range(self.num_epochs):
            predicted_y = []
            for i in range(X.shape[0]):
                predicted_y.append(self.__evreg_predict(X, y, i, cur_weights, self.p, self.k, self.radius))
            cur_loss = self.__count_loss(y, predicted_y)
            cur_weights = self.__update_weights(X, y, self.alpha, cur_weights, self.p, self.radius)
            if cur_loss < min_loss:
                best_weights = cur_weights
        cutting_rule = apply_cr(self.cutting_rule)

        self.selected_features_ = cutting_rule(dict(zip(range(1, len(best_weights)), best_weights)))
