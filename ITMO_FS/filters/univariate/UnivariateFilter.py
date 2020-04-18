from sklearn.base import TransformerMixin

from .measures import GLOB_CR, GLOB_MEASURE


class UnivariateFilter(TransformerMixin):  # TODO ADD LOGGING
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

    def _check_input(self, x, y, feature_names):
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(y, 'values'):
            y = y.values
        self.feature_scores = None

        if hasattr(x, 'columns'):
            feature_names = x.columns
        else:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))

        return x, y, feature_names

    def get_scores(self, x, y, feature_names=None):
        x, y, feature_names = self._check_input(x, y, feature_names)
        return dict(zip(feature_names, self.measure(x, y)))

    def fit_transform(self, X, y=None, feature_names=None, store_scores=False, **fit_params):
        self.fit(X, y, feature_names, store_scores)
        return self.transform(X)

    def fit(self, x, y, feature_names=None, store_scores=True):
        x, y, feature_names = self._check_input(x, y, feature_names)
        feature_scores = self.get_scores(x, y, feature_names)

        if store_scores:
            self.feature_scores = feature_scores
        self.selected_features = self.cutting_rule(feature_scores)

    def transform(self, x):
        return x[:, self.selected_features]
