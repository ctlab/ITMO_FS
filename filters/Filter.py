import filters.FitCriterion  # TODO: .run() feature_names
import filters.GiniIndexFilter
import filters.IGFilter  # TODO: strange .run() interface; .run() feature_names; no default constructor
import filters.RandomFilter  # TODO: bad .run() interface; .run() feature_names; no default constructor
# TODO: move all feature_names?

# Default-constructed measures
class DefaultMeasures:
    FitCriterion = filters.FitCriterion()  # Can be customized
    GiniIndex = filters.GiniIndexFilter()
    # IGFilter = filters.IGFilter()
    # RandomFilter = filters.RandomFilter()


GLOB_MEASURE = {"FitCriterion": DefaultMeasures.FitCriterion}

GLOB_CR = {}

class DefaultCuttingRules:
    pass

class Filter:
    def __init__(self, measure, cutting_rule):
        self.measure = measure
        self.cutting_rule = cutting_rule
        self.feature_scores = None

    def run(self, x, y, store_scores=False):
        self.feature_scores = None
        feature_scores = self.measure.run(x, y)
        if store_scores:
            self.feature_scores = feature_scores
        selected_features = self.cutting_rule(feature_scores)
        #selected_features




