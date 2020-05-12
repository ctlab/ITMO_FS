from sklearn.base import TransformerMixin

from .measures import GLOB_CR, GLOB_MEASURE


class UnivariateFilter(TransformerMixin):  # TODO ADD LOGGING
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

        from sklearn.datasets import make_classification
        from ITMO_FS.filters.univariate import select_k_best
        from ITMO_FS.filters.univariate import UnivariateFilter
        from ITMO_FS.filters.univariate import f_ratio_measure

        x, y = make_classification(1000, 100, n_informative = 10, n_redundant = 30, n_repeated = 10, shuffle = False)
        ufilter = UnivariateFilter(f_ratio_measure, select_k_best(10))
        ufilter.fit(x, y)
        print(ufilter.selected_features)
    """

    def __init__(self, measure, cutting_rule):
        # TODO Check measure and cutting_rule
        if type(measure) is str:
            try:
                self.measure = GLOB_MEASURE[measure]
            except KeyError:
                raise KeyError("No %r measure yet" % measure)
        else:
            self.measure = measure

        if type(cutting_rule) is str:
            try:
                self.cutting_rule = GLOB_CR[cutting_rule]
            except KeyError:
                raise KeyError("No %r cutting rule yet" % measure)
        else:
            self.cutting_rule = cutting_rule
        self.feature_scores = None
        self.hash = None
        self.selected_features = None

    def _check_input(self, X, y, feature_names):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            # TODO Fix case of y passed as DataFrame. For now y is transformed to 2D array and this causes an error.
            #  It seems better to follow usual sklearn practice using check_X_y but y = y[0].values is also possible
            y = y.values

        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            if feature_names is None:
                feature_names = list(range(X.shape[1]))

        return X, y, feature_names

    def get_scores(self, X, y, feature_names=None):
        """
            Counts feature scores on given data.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ------
            dictionary of format: key - feature_names if defined or number of features,
            values - feature scores

        """
        X, y, feature_names = self._check_input(X, y, feature_names)
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
        feature_scores = self.get_scores(X, y, feature_names)

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
        return X[:, self.selected_features]
