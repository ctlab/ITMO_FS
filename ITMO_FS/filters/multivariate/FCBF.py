import numpy as np

from ...utils.information_theory import entropy, conditional_entropy
from ...utils import BaseTransformer, generate_features


class FCBFDiscreteFilter(BaseTransformer):
    """
    Creates FCBF (Fast Correlation Based filter) feature selection filter
    based on mutual information criteria for data with discrete features
    This filter finds best set of features by searching for a feature,
    which provides the most information about classification problem on
    given dataset at each step and then eliminating features which are less
    relevant than redundant

        Parameters
        ----------
        delta : float
            Symmetric uncertainty value threshold.

        Notes ----- For more details see `this paper
        <https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf/>`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]], dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> fcbf = FCBFDiscreteFilter().fit(X, y)
        >>> fcbf.selected_features_
        array([4], dtype=int64)
    """

    def __init__(self, delta=0.1):
        self.delta = delta

    def _fit(self, x, y, **kwargs):
        """
            Fits filter

            Parameters
            ----------

            x : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                The target values.
            **kwargs
            Returns
            -------
            None
        """

        def __SU(x, y, entropy_y):
            entropy_x = entropy(x)
            return 2 * (entropy_x - conditional_entropy(y, x)) / (
                entropy_x + entropy_y)

        free_features = generate_features(x)
        self.selected_features_ = np.array([], dtype='int')
        entropy_y = entropy(y)
        su_class = np.apply_along_axis(__SU, 0, x, y, entropy_y)
        self.selected_features_ = np.argsort(su_class)[::-1][:np.count_nonzero(su_class > self.delta)]
        index = 1
        while index < self.selected_features_.shape[0]:
            feature = self.selected_features_[index - 1]
            entropy_feature = entropy(x[:, feature])
            su_classes = su_class[self.selected_features_[index:]]
            su_feature = np.apply_along_axis(__SU, 0, x[:, self.selected_features_[index:]],
                x[:, feature], entropy_feature)
            self.selected_features_ = np.delete(self.selected_features_, np.flatnonzero(su_feature >= su_classes) + index)
            index += 1
