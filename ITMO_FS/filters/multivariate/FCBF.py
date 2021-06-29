from logging import getLogger

import numpy as np

from ...utils import BaseTransformer, generate_features
from ...utils.information_theory import entropy, conditional_entropy


class FCBFDiscreteFilter(BaseTransformer):
    """Create FCBF (Fast Correlation Based filter) feature selection filter
    based on mutual information criteria for data with discrete features. This
    filter finds best set of features by searching for a feature, which
    provides the most information about classification problem on given dataset
    at each step and then eliminating features which are less relevant than
    redundant.

    Parameters
    ----------
    delta : float
        Symmetric uncertainty value threshold.

    Notes
    -----
    For more details see `this paper
    <https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> fcbf = FCBFDiscreteFilter().fit(X, y)
    >>> fcbf.selected_features_
    array([4], dtype=int64)
    """
    def __init__(self, delta=0.1):
        self.delta = delta

    def _fit(self, x, y):
        """Fit the filter.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        def __SU(x, y, entropy_y):
            entropy_x = entropy(x)
            return 2 * ((entropy_x - conditional_entropy(y, x))
                        / (entropy_x + entropy_y))

        free_features = generate_features(x)
        self.selected_features_ = np.array([], dtype='int')
        entropy_y = entropy(y)
        getLogger(__name__).info("Entropy of y: %f", entropy_y)

        su_class = np.apply_along_axis(__SU, 0, x, y, entropy_y)
        getLogger(__name__).info("SU values against y: %s", su_class)
        self.selected_features_ = np.argsort(su_class)[::-1][:
            np.count_nonzero(su_class > self.delta)]
        getLogger(__name__).info("Selected set: %s", self.selected_features_)

        index = 1
        while index < self.selected_features_.shape[0]:
            feature = self.selected_features_[index - 1]
            getLogger(__name__).info("Leading feature: %d", feature)
            entropy_feature = entropy(x[:, feature])
            getLogger(__name__).info(
                "Leading feature entropy: %f", entropy_feature)
            su_classes = su_class[self.selected_features_[index:]]
            getLogger(__name__).info(
                "SU values against y for the remaining features: %s",
                su_classes)
            su_feature = np.apply_along_axis(
                __SU, 0, x[:, self.selected_features_[index:]], x[:, feature],
                entropy_feature)
            getLogger(__name__).info(
                "SU values against leading feature for the remaining features: "
                "%s", su_feature)
            to_delete = np.flatnonzero(su_feature >= su_classes) + index
            getLogger(__name__).info(
                "Deleting those features from the selected set: %s",
                self.selected_features_[to_delete])
            self.selected_features_ = np.delete(
                self.selected_features_, to_delete)
            index += 1
