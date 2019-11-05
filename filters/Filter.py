from .utils import *


class Filter(object):####TODO add logging
    def __init__(self, measure, cutting_rule):
        """
        Basic univariate filter class with chosen(even custom) measure and cutting rule
        :param measure:
            Examples
         --------
        >>> f=Filter("PearsonCorr", GLOB_CR["K best"](6))
        """
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

    def run(self, x, y, feature_names=None, store_scores=False, verbose=0):
        try:
            x = x.values
            y = y.values
        except AttributeError:
            x = x
        self.feature_scores = None
        try:
            feature_names = x.columns
        except AttributeError:
            if feature_names is None:
                feature_names = list(range(x.shape[1]))
        feature_scores = None
        if not (self.hash == hash(self.measure)):
            feature_scores = dict(zip(feature_names, self.measure(x, y)))
            self.hash = hash(self.measure)

        if store_scores:
            self.feature_scores = feature_scores
        selected_features = self.cutting_rule(feature_scores)
        return x[:, selected_features]
