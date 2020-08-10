from numpy import ndarray
from sklearn.base import TransformerMixin, BaseEstimator

from .measures import GLOB_CR, GLOB_MEASURE
from ...utils import DataChecker, generate_features, check_restrictions


class UnivariateFilter(BaseEstimator, TransformerMixin, DataChecker):  # TODO ADD LOGGING
    """
        Basic interface for using univariate measures for feature selection.
        List of available measures is in ITMO_FS.filters.univariate.measures, also you can
        provide your own measure but it should suit the argument scheme for measures,
        i.e. take two arguments x,y and return scores for all the features in dataset x.
        Same applies to cutting rules.

        Parameters
        ----------
        measure : string or callable
            A metric name defined in GLOB_MEASURE or a callable with signature
            measure (sample dataset, labels of dataset samples)
            which should return a list of metric values for each feature in the dataset.
        cutting_rule : string or callables
            A cutting rule name defined in GLOB_CR or a callable with signature
            cutting_rule (features),
            which should return a list features ranked by some rule.

        See Also
        --------

        Examples
        --------

        >>> from sklearn.datasets import make_classification
        >>> from ITMO_FS.filters.univariate import select_k_best
        >>> from ITMO_FS.filters.univariate import UnivariateFilter
        >>> from ITMO_FS.filters.univariate import f_ratio_measure
        >>> x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, \
n_repeated = 10, shuffle = False)
        >>> ufilter = UnivariateFilter(f_ratio_measure, select_k_best(10))
        >>> ufilter.fit(x, y)
        >>> print(ufilter.selected_features)
    """

    def __init__(self, measure, cutting_rule=("Best by percentage", 0.2)):
        # TODO Check measure and cutting_rule
        if type(measure) is str:
            try:
                self.measure = GLOB_MEASURE[measure]
            except KeyError:
                raise KeyError("No %r measure yet" % measure)
        elif hasattr(measure, '__call__'):
            self.measure = measure
        else:
            raise KeyError("%r isn't a measure function or string" % measure)

        if type(cutting_rule) is tuple:
            cutting_rule_name = cutting_rule[0]
            cutting_rule_value = cutting_rule[1]
            try:
                self.cutting_rule = GLOB_CR[cutting_rule_name](cutting_rule_value)
            except KeyError:
                raise KeyError("No %r cutting rule yet" % cutting_rule_name)
        elif hasattr(cutting_rule, '__call__'):
            self.cutting_rule = cutting_rule
        else:
            raise KeyError("%r isn't a cutting rule function or string" % cutting_rule)

        check_restrictions(self.measure.__name__, self.cutting_rule.__name__)

    def get_scores(self, X, y, feature_names):
        """
            Counts feature scores on given data.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings
                In case you want to define feature names

            Returns
            ------
            dictionary of format: key - feature_names, values - feature scores

        """
        return dict(zip(feature_names, self.measure(X, y)))

    def fit_transform(self, X, y=None, feature_names=None, store_scores=False, **fit_params):
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
            store_scores : boolean, optional (by default False)
                In case you want to store the scores of features
                for future calls to Univariate filter
            **fit_params :
                dictonary of measure parameter if needed.

            Returns
            ------
            X dataset sliced with features selected by the filter
        """
        self.fit(X, y, feature_names, store_scores)
        return self.transform(X)

    def fit(self, X, y, feature_names=None, store_scores=True):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names
            store_scores : boolean, optional (by default False)
                In case you want to store the scores of features
                for future calls to Univariate filter

            Returns
            ------
            None
        """
        X, y, feature_names = self._check_input(X, y, feature_names)
        features = generate_features(X, feature_names)
        self.feature_names = dict(zip(features, feature_names))
        feature_scores = self.get_scores(X, y, features)

        if store_scores:
            self.feature_scores = feature_scores
        self.selected_features = self.cutting_rule(feature_scores)

    def transform(self, X):
        """
            Slices given dataset by previously selected features.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.

            Returns
            ------
            X dataset sliced with features selected by the filter
        """
        if type(X) is ndarray:
            return X[:, self.selected_features]
        else:
            return X[self.selected_features]

    def __repr__(self):
        return "Univariate filter with measure {} and cutting rule {}".format(self.measure.__name__,
                                                                              self.cutting_rule.__name__)
