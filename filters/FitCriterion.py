import numpy as np
from utils.data_check import generate_features

from utils import generate_features


class FitCriterion:
    """
        Creates Fit Criterion builder
        https://link.springer.com/chapter/10.1007/978-3-642-14400-4_20

        Parameters
        ----------
        mean: function
            Function used to find center of set of points.
            np.median could be also suggested for this role.

        See Also
        --------

        Examples
        --------
        x = np.array([[4, 1, 3, 2, 5],
                      [5, 4, 3, 1, 4],
                      [5, 2, 3, 0, 5],
                      [1, 1, 4, 0, 5]])
        y = np.array([2,
                      1,
                      0,
                      0])
        fc = FitCriterion()
        fc.run(x, y)
        {0: 0.75, 1: 0.75, 2: 0.5, 3: 1.0, 4: 0.75}
    """

    def __init__(self, mean=np.mean):
        self.mean = mean

    feature_scores = {}

    def run(self, x, y, feature_names=None):
        """
            Parameters
            ----------
            x: array-like, shape (n_features, n_samples)
                Input samples' parameters.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential integers.
            feature_names: iterable
                Names for features in resulting dict, sequential integers will be used if None passed.

            Returns
            -------
            result: python dict containing entries (feature, ratio)
                Dictionary with Fit Criterion ratios for input dataset
                If passed data has no `.features' field, feature labels will be generated as sequential integers
                starting from zero.

            See Also
            --------
            utils.data_check.generate_features()

            Examples
            --------
            :param feature_names: names for features, not needed for pandas DataFrames
        """
        feature_names = generate_features(x, feature_names)  # Generating feature labels for output data

        x = np.asarray(x)  # Converting input data to numpy array
        y = np.asarray(y)

        fc = np.zeros(x.shape[1])  # Array with amounts of correct predictions for each feature

        tokensN = np.max(y) + 1  # Number of different class tokens
        feature_names = generate_features(x, feature_names)
        # Utility arrays
        centers = np.empty(tokensN)  # Array with centers of sets of feature values for each class token
        variances = np.empty(tokensN)  # Array with variances of sets of feature values for each class token
        # Each of arrays above will be separately calculated for each feature

        distances = np.empty(tokensN)  # Array with distances between sample's value and each class's center
        # This array will be separately calculated for each feature and each sample

        for feature_index, feature in enumerate(x.T):  # For each feature
            # Initializing utility structures
            class_values = [[] for _ in range(tokensN)]  # Array with lists of feature values for each class token
            for index, value in enumerate(y):  # Filling array
                class_values[value].append(feature[index])
            for token, values in enumerate(class_values):  # For each class token's list of feature values
                tmp_arr = np.array(values)
                centers[token] = self.mean(tmp_arr)
                variances[token] = np.var(tmp_arr)

            # Main calculations
            for sample_index, value in enumerate(feature):  # For each sample value
                for i in range(tokensN):  # For each class token
                    # Here can be raise warnings by 0/0 division. In this case, default results
                    # are interpreted correctly
                    distances[i] = np.abs(value - centers[i]) / variances[i]
                fc[feature_index] += np.argmin(distances) == y[sample_index]

        fc /= y.shape[0]  # Normalization

        self.feature_scores = dict(zip(feature_names, fc))
        return self.feature_scores

    def __repr__(self):
        return "Fit criterion with mean {}".format(self.mean)