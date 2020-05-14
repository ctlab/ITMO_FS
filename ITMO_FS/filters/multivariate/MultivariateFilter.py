import numpy as np

from .measures import GLOB_MEASURE
from ...utils import generate_features


# TODO X and y transformation for DataFrame support
# TODO Test interface!!!!
class MultivariateFilter(object):
    """
        Provides basic functionality for multivariate filters.
        Parameters
        ----------
        measure : string or callable
            A metric name defined in GLOB_MEASURE or a callable with signature measure(selected_features, free_features, dataset, labels)
            which should return a list of metric values for each feature in the dataset.
        n_features : int
            Number of features to select.
        beta : float, optional
            Initialize only in case you run MIFS or generalizedCriteria metrics.
            Initialize only in case you run MIFS or generalizedCriteria metrics
        gamma : float, optional
            Initialize only in case you run generalizedCriteria metric.

            Initialize only in case you run eneralizedCriteria metric
        
        See Also
        --------
        
        Examples
        --------
        from ITMO_FS.filters.multivariate import MultivariateFilter
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import KBinsDiscretizer

        import numpy as np

        dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data, target = np.array(dataset[0]), np.array(dataset[1])
        est.fit(data)
        data = est.transform(data)
        model = MultivariateFilter('MRMR', 8)
        model.fit(data, target)
        print(model.selected_features)
    """

    def __init__(self, measure, n_features, beta=None, gamma=None):
        if type(measure) is str:
            try:
                self.measure = GLOB_MEASURE[measure]
            except KeyError:
                raise KeyError("No %r measure yet" % measure)
        else:
            self.measure = measure
        self.__n_features = n_features
        self.selected_features = np.array([], dtype=np.integer)
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

            Returns
            ------
            None

        """
        if self.__n_features > X.shape[1]:
            raise ValueError("Cannot select %d features out of %d" % (self.__n_features, X.shape[1]))
        values = np.array([])
        free_features = generate_features(X)
        while len(self.selected_features) != self.__n_features:
            if self.beta is None:
                values = self.measure(self.selected_features, free_features, X, y)
            else:
                if self.gamma is not None:
                    values = self.measure(self.selected_features, free_features, X, y, self.beta, self.gamma)
                else:
                    values = self.measure(self.selected_features, free_features, X, y, self.beta)
            to_add = np.argmax(values)
            self.selected_features = np.append(self.selected_features, free_features[to_add])
            free_features = np.delete(free_features, to_add)

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            
            Returns
            ------
            
            Transformed 2D numpy array

        """

        return X[:, self.selected_features]
