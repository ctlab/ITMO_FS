import filters
# TODO: move all feature_name-s?


class DefaultMeasures:
    FitCriterion = filters.FitCriterion()  # Can be customized

    GiniIndex = filters.GiniIndexFilter()

    # IGFilter = filters.IGFilter()  # TODO: unexpected .run() interface; .run() feature_names; no default constructor

    RandomFilter = filters.RandomFilter()  # Can be customized. Be careful, random is determined by previous
    # operations on this instance of filter. See more information in filter's definition. Use "upper_border" cutting
    # rule for further feature selection.

    SpearmanCorrelation = filters.SpearmanCorrelationFilter()

    # SymmetricUncertainty = filters.SymmetricUncertainty()  # TODO: not implemented yet

    # VDM = filters.VDM()  # TODO: probably not a filter


GLOB_MEASURE = {"fit": DefaultMeasures.FitCriterion,  # TODO: put finished filters here
                "gini": DefaultMeasures.GiniIndex,
                "spearman": DefaultMeasures.FitCriterion,
                "random": DefaultMeasures.RandomFilter}


class DefaultCuttingRules:
    @staticmethod
    def top_k(k):
        """
            Selects k features with biggest features scores.

            Parameters
            ----------
            k: integer
                Number of features

            Returns
            -------
            result: function
                Function which takes dict of kind {feature_name: feature_score} and returns list with k
                feature_name-s with biggest feature_score-s.

            See Also
            --------

        """
        def impl(feature_scores):
            selected_fs = sorted(feature_scores.items(), key=lambda x: x[1])[-k:]
            return list(map(lambda x: x[0], selected_fs))
        return impl

    @staticmethod
    def filter_cr(predicate):
        """
            Selects features which satisfy passed predicate.

            Parameters
            ----------
            predicate: function
                Function, which takes feature_score and returns bool.

            Returns
            -------
            result: function
                Function which takes dict of kind {feature_name: feature_score} and returns list with feature_name-s
                which satisfy passed predicate.

            See Also
            --------

        """
        def impl(feature_scores):
            return [feature for feature, score in feature_scores.items() if predicate(score)]
        return impl

    @staticmethod
    def border(value, include=True):
        """
            Selects features with feature_value higher (or equal, if selected) to passed border value.

            Parameters
            ----------
            value: number
                Border value, features with scores below this border will be discarded
            include: bool
                Specifies inclusion of border value

            Returns
            -------
            result: function
                Function which takes dict of kind {feature_name: feature_score} and returns list with feature_name-s
                with feature_value-s higher or equal to border value.

            See Also
            --------

        """
        if include:
            return DefaultCuttingRules.filter_cr(lambda x: x >= value)
        else:
            return DefaultCuttingRules.filter_cr(lambda x: x > value)

    @staticmethod
    def upper_border(value, include=False):
        """
            Selects features with feature_value less (or equal, if selected) to passed border value.

            Parameters
            ----------
            value: number
                Border value, features with scores above this border will be discarded
            include: bool
                Specifies inclusion of border value

            Returns
            -------
            result: function
                Function which takes dict of kind {feature_name: feature_score} and returns list with feature_name-s
                with feature_value-s less or equal to border value.

            See Also
            --------

        """
        if include:
            return DefaultCuttingRules.filter_cr(lambda x: x <= value)
        else:
            return DefaultCuttingRules.filter_cr(lambda x: x < value)


GLOB_CR = {"best": DefaultCuttingRules.top_k,
           "filter": DefaultCuttingRules.filter_cr,
           "border": DefaultCuttingRules.border,
           "upper_border": DefaultCuttingRules.upper_border}


class Filter:
    """
        Interface for using filters.

        Parameters
        ----------
        measure:
            1) str
            One of default-constructed measures from GLOB_MEASURE
            2) class
            Filter class instance with .run() method
        cutting_rule:
            1) str
            One of suggested cutting rules from GLOB_CR
            2) function
            Custom cutting rule which takes dict of kind {feature_name: feature_score} and returns list with selected
            feature_name-s
        parameter:
            Parameter for initializing string-based cutting rule. Should be specified if and only if string-based
            cutting rule is used.

        See Also
        --------

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([[0, 0, 0, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 2]])
        >>> y = np.array([0,
        ...               1,
        ...               1])
        >>> gini_filter = Filter("gini", "border", 0.5)  # Filter which drops features with Gini Index less than 0.5
        >>> gini_filter.run(x, y, stores_scores=True)
        array([[0 0]
               [1 1]
               [0 2]])
        >>> gini_filter.feature_scores  # Can be accessed if `store_scores` flag was set True on the last .run() call.
        {0: 0.0, 1: nan, 2: 0.5, 3: 0.16666666666666674}
        >>>
        >>> import pandas as pd
        >>> x_pandas = pd.DataFrame(x)  # Pandas example
        >>> x_pandas.rename(columns={0: "a", 1: "b", 2: "c", 3: "d"})  # Setting names to columns.
        >>> y_pandas = pd.Series(y)
        >>> rand_filter = Filter("random", "upper_border", 2)  # Filter that randomly chooses 2 features
        >>> rand_filter.run(x_pandas, y_pandas)
           c  d
        0  0  0
        1  1  1
        2  0  2
        >>>
        >>> my_filter = Filter(filters.FitCriterion(mean=np.median), DefaultCuttingRules.top_k(2))  # Filter which takes
        # two best features based on customized Fit Criterion measure
        >>> my_filter.run(x, y, store_scores=True)
        array([[0 0]
               [1 1]
               [1 2]])
    """
    def __init__(self, measure, cutting_rule, parameter=None):
        if type(measure) is str:
            try:
                self.measure = GLOB_MEASURE[measure]
            except KeyError:
                raise KeyError("Unexpected string-based measure selection")
        else:
            self.measure = measure

        if type(cutting_rule) is str:
            if parameter is None:
                raise ValueError("Parameter for string-based cutting rule tune-up expected")
            try:
                self.cutting_rule = GLOB_CR[cutting_rule](parameter)
            except KeyError:
                raise KeyError("Unexpected string-based cutting rule selection")
        else:
            self.cutting_rule = cutting_rule
        self.feature_scores = None

    def run(self, x, y, store_scores=False):  # If `y` parameter is not necessary for specified filter, None can be
        # passed.
        self.feature_scores = None
        feature_scores = self.measure.run(x, y)
        if store_scores:
            self.feature_scores = feature_scores
        selected_features = self.cutting_rule(feature_scores)
        if not hasattr(x, "loc"):  # numpy.ndarray/list
            return x[:, selected_features]
        else:  # pandas.DataFrame
            return x.loc[:, selected_features]
