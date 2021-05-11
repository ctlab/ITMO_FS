from .measures import CR_NAMES, MEASURE_NAMES
import numpy as np
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

        >>> import numpy as np
        >>> from ITMO_FS.filters.univariate import select_k_best
        >>> from ITMO_FS.filters.univariate import UnivariateFilter
        >>> from ITMO_FS.filters.univariate import f_ratio_measure
        >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], \
            [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
        >>> y = np.array([1, 3, 2, 1, 2])
        >>> filter = UnivariateFilter(f_ratio_measure, \
            select_k_best(2)).fit(x, y)
        >>> filter.selected_features_
        array([4, 2], dtype=int64)
        >>> filter.feature_scores_
        array([0.6 , 0.2 , 1.  , 0.12, 5.4 ])
    """

    def __init__(self, measure, cutting_rule=("Best by percentage", 1.0)):
        self.measure = measure
        self.cutting_rule = cutting_rule

    def __apply_ms(self):
        if isinstance(self.measure, str):
            try:
                measure = MEASURE_NAMES[self.measure]
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

        feature_scores = measure(X, y)

        if store_scores:
            self.feature_scores_ = feature_scores
        self.selected_features_ = cutting_rule(feature_scores)
