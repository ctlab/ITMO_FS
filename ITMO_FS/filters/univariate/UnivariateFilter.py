from numpy import ndarray

from .measures import GLOB_CR, GLOB_MEASURE
from ...utils import BaseTransformer, generate_features, check_restrictions, \
    apply_cr


class UnivariateFilter(BaseTransformer):  # TODO ADD LOGGING
    """
    Basic interface for using univariate measures for feature selection.
    List of available measures is in ITMO_FS.filters.univariate.measures,
    also you can provide your own measure but it should suit the argument
    scheme for measures, i.e. take two arguments x,y and return scores for
    all the features in dataset x. Same applies to cutting rules.

        Parameters
        ----------
        measure : string or callable
            A metric name defined in GLOB_MEASURE or a callable with signature
            measure (sample dataset, labels of dataset samples) which should
            return a list of metric values for each feature in the dataset.
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
        >>> print(ufilter.selected_features_)
    """

    def __init__(self, measure, cutting_rule=("Best by percentage", 1.0)):
        super().__init__()
        self.measure = measure
        self.cutting_rule = cutting_rule

    def __apply_ms(self):
        if isinstance(self.measure, str):
            try:
                measure = GLOB_MEASURE[self.measure]
            except KeyError:
                raise KeyError("No %r measure yet" % self.measure)
        elif hasattr(self.measure, '__call__'):
            measure = self.measure
        else:
            raise KeyError(
                "%r isn't a measure function or string" % self.measure)
        return measure

    def _fit(self, X, y, store_scores=True):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            store_scores : boolean, optional
                In case you want to store the scores of features
                for future calls to Univariate filter; default False

            Returns
            ------
            None
        """

        measure = self.__apply_ms()
        cutting_rule = apply_cr(self.cutting_rule)

        check_restrictions(measure.__name__, cutting_rule.__name__)

        features = generate_features(X)
        feature_scores = dict(zip(features, measure(X, y)))

        if store_scores:
            self.feature_scores_ = feature_scores
        self.selected_features_ = cutting_rule(feature_scores)
