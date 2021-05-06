import numpy as np
from sklearn.base import TransformerMixin

from .measures import MEASURE_NAMES, mutual_information, \
    matrix_mutual_information
from ...utils import BaseTransformer, generate_features


# TODO Test interface!!!!


class MultivariateFilter(BaseTransformer):
    """
        Provides basic functionality for multivariate filters.

        Parameters
        ----------
        measure : string or callable
            A metric name defined in GLOB_MEASURE or a callable with signature
            measure(selected_features, free_features, dataset, labels) which
            should return a list of metric values for each feature
            in the dataset.
        n_features : int
            Number of features to select.
        beta : float, optional
            Initialize only in case you run MIFS or generalizedCriteria
            metrics.
        gamma : float, optional
            Initialize only in case you run generalizedCriteria metric.

        See Also
        --------

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import MultivariateFilter
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> import numpy as np

        >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> data = est.fit_transform(X)
        >>> model = MultivariateFilter('JMI', 3).fit(X, y)
        >>> model.selected_features_
        array([4, 0, 1])
    """

    def __init__(self, measure, n_features, beta=None, gamma=None):
        self.measure = measure
        self.n_features = n_features
        self.beta = beta
        self.gamma = gamma

    def _fit(self, X, y, **kwargs):
        """
            Fits the filter.

            Parameters
            ----------

            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.
            **kwargs
            Returns
            ------
            None
        """

        if isinstance(self.measure, str):
            try:
                self.measure = MEASURE_NAMES[self.measure]
            except KeyError:
                raise KeyError("No %r measure yet" % self.measure)

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" %
                (self.n_features, self.n_features_))
        free_features = generate_features(X)
        if "selected_features" in kwargs:
            self.selected_features_ = kwargs["selected_features"]
        else:
            self.selected_features_ = np.array([], dtype='int')
        relevance = np.apply_along_axis(mutual_information, 0,
                                        X[:, free_features],
                                        y)

        redundancy = np.vectorize(
            lambda free_feature: 
                matrix_mutual_information(X[:, free_features],
                                          X[:, free_feature]),
                signature='()->(1)')(
            free_features)

        while len(self.selected_features_) != self.n_features:
            if self.beta is None:
                values = self.measure(
                    self.selected_features_, free_features, X, y,
                    relevance=relevance[free_features],
                    redundancy=np.sum(redundancy[self.selected_features_], axis=0)[free_features])
            else:
                if self.gamma is not None:
                    values = self.measure(
                        self.selected_features_,
                        free_features,
                        X,
                        y,
                        self.beta,
                        self.gamma, 
                        relevance=relevance[free_features], 
                        redundancy=np.sum(redundancy[self.selected_features_], axis=0)[free_features])
                else:
                    values = (self.measure(
                        self.selected_features_,
                        free_features,
                        X,
                        y,
                        self.beta, 
                        relevance=relevance[free_features], 
                        redundancy=np.sum(redundancy[self.selected_features_], axis=0)[free_features])
                    )
            print(values)
            to_add = np.argmax(values)
            self.selected_features_ = np.append(
                self.selected_features_, free_features[to_add])
            free_features = np.delete(free_features, to_add)
