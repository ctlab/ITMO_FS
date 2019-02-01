import random as rnd


class RandomFilter:
    """
        Creates random feature selector

        Parameters
        ----------
        n_features : int
            Number of features to select.
        seed: int
            Seed for python random.

        See Also
        --------


        Examples
        --------
        >>> filter = RandomFilter(2,8)
        >>> filter.run(np.ones((5, 10))))
        [3, 6]
        """

    def __init__(self, n_features, seed=1):
        self.n_features_ = n_features
        rnd.seed(seed)

    def run(self, X):
        """
            Runs feature selector

            Parameters
            ----------
            X : array-like, shape (n_features,n_samples)
                Number of features to select.
            Returns
            ------
            result: list
                List of selected features
            See Also
            --------
            Examples
            --------
            >>> filter = RandomFilter(2,8)
            >>> filter.run(np.ones((5, 10))))
            [3, 6]
        """
        result = []
        rest_features = list(range(len(X)))
        for i in range(self.n_features_):
            ind = rnd.randint(0, len(rest_features) - 1)
            result.append(rest_features[ind])
            del rest_features[ind]
        return result
