import numpy as np
from utils.data_check import generate_features


class RandomFilter:
    """
        Creates random feature selector

        Parameters
        ----------
        seed: int
            Seed for python random.

        See Also
        --------


        Examples
        --------
        #>>> filter = RandomFilter(2,8)
        #>>> filter.run(np.ones((5, 10))))
        [3, 6]
        """

    def __init__(self, seed=1):
        self.rng = np.random.RandomState(seed)

    def run(self, x, y=None, feature_names=None):
        """
            Parameters
            ----------
            x: array-like, shape (n_features, n_samples)
                Input samples' parameters.
            y: object
                Input samples' class labels. Unnecessary parameter here, need here only for right interface.
            feature_names: iterable
                Names for features in resulting dict, sequential integers will be used if None passed.

            Returns
            -------
            result: python dict containing entries (feature, score)
                Dictionary with scores for input dataset. Scores is a random permutation of indexes in range of
                n_features. Random determined by both of:
                1) passed seed
                2) all previous calls of this method

            See Also
            --------

            Examples
            --------
        """
        feature_names = generate_features(x, feature_names)  # Generating feature labels for output data

        x = np.asarray(x)  # Converting input data to numpy array

        scores = self.rng.permutation(x.shape[1])

        return dict(zip(feature_names, scores))
