import numpy as np
from sklearn.base import TransformerMixin

from .measures import GLOB_MEASURE
from ...utils import DataChecker, generate_features

# TODO Test interface!!!!
class MultivariateFilter(TransformerMixin, DataChecker):
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
        gamma : float, optional
            Initialize only in case you run generalizedCriteria metric.
        
        See Also
        --------
        
        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import MultivariateFilter
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20, \
n_informative=4, n_redundant=0, shuffle=False)
        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> data, target = np.array(dataset[0]), np.array(dataset[1])
        >>> est.fit(data)
        >>> data = est.transform(data)
        >>> model = MultivariateFilter('MRMR', 8)
        >>> model.fit(data, target)
        >>> print(model.selected_features)
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
        self.selected_features = np.array([], dtype='int')
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y, feature_names=None):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ------
            None
        """

        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)
        if self.__n_features > X.shape[1]:
            raise ValueError("Cannot select %d features out of %d" % (self.__n_features, X.shape[1]))
        free_features = generate_features(X)
        self.feature_names = dict(zip(features, feature_names))
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
        self.selected_features = features[self.selected_features]

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

        if type(X) is np.ndarray:
            return X[:, self.selected_features]
        else:
            return X[self.selected_features]

    def fit_transform(self, X, y=None, feature_names=None, **fit_params):
        """
            Fits the filter and transforms given dataset X.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, ), optional
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names
            **fit_params :
                dictonary of measure parameter if needed.

            Returns
            ------

            X dataset sliced with features selected by the filter
        """

        self.fit(X, y, feature_names)
        return self.transform(X)
