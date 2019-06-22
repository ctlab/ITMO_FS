import filters.FitCriterion
import filters.GiniIndexFilter


# Default-constructed measures
class DefaultMeasures:
    FitCriterion = filters.FitCriterion()
    GiniIndex = filters.GiniIndexFilter()

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




