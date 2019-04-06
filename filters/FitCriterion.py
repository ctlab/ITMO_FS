import numpy as np
from utils.data_check import generate_features

# TODO: Special cases(see below)
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
        >>> x = np.array([[4, 1, 3, 2, 5],
        ...               [5, 4, 3, 1, 4],
        ...               [5, 2, 3, 0, 5],
        ...               [1, 1, 4, 0, 5]])
        >>> y = np.array([2,
        ...               1,
        ...               0,
        ...               0])
        >>> fc = FitCriterion()
        >>> fc.run(x, y)
        {0: 0.75, 1: 0.75, 2: 0.5, 3: 1.0, 4: 0.75}

    """
    def __init__(self, mean=np.mean):
        self.mean = mean

    def run(self, x, y):
        """
            Parameters
            ----------
            x: array-like, shape (n_features, n_samples)
                Input samples' parameters.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential integers.

            Returns
            -------
            result:
                numpy.ndarray, shape (n_features) with Fit Criterion ratios for each feature

            See Also
            --------

            Examples
            --------
        """
        feature_labels = generate_features(x, None)  # Generating feature labels for output data
        x = np.asarray(x)  # Converting input data to numpy array
        y = np.asarray(y)

        fc = np.zeros(x.shape[1])  # Array with amounts of correct predictions for each feature
        tokensN = int(np.max(y)) + 1  # Number of different class tokens

        # Utility arrays
        centers = np.empty(tokensN)  # Array with centers of sets of feature values for each class token
        variances = np.empty(tokensN)  # Array with variances of sets of feature values for each class token
        # Each of arrays above will be separately calculated for each feature
        distances = np.empty(tokensN)  # Array with distances between sample's value and each class's center
        # This array will be separately calculated for each feature and each sample

        for feature_index in range(x.shape[1]):  # For each feature
            feature = x.T[feature_index]
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
                    # TODO: cases 0 / 0, smth / 0
                    # TODO: disable warnings
                    distances[i] = np.abs(value - centers[i]) / variances[i]
                fc[feature_index] += np.argmin(distances) == y[sample_index]

        fc /= y.shape[0]  # Normalization

        return dict(zip(feature_labels, fc))  # Adding feature labels
