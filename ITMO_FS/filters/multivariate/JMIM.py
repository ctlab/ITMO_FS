import numpy as np
from ITMO_FS.utils.information_theory import matrix_mutual_information
from ITMO_FS.utils.information_theory import joint_mutual_information
from ITMO_FS.utils.information_theory import symmetrical_relevance
from ...utils import BaseTransformer, generate_features


class JMIM(BaseTransformer):
    """
    Creates JMIM (Joint Mutual Information Maximisation) feature selection
    filter based on joint mutual information criterion for data with
    discrete features. This filter finds best set of features by maximasing
    joint mutual information for a candidate feature on its combination with
    target variable and each of the already selected features separately.
    Normalized JMIM uses normalised joint mutual information as a criterion.

        Parameters
        ----------
        n_features : int
            Number of features to select.
        normalized : bool
            Whether to use normalized version of JMIM or not.


        Notes ----- For more details see `this paper
        <https://www.sciencedirect.com/science/article/pii/S0957417415004674
        >`_.

        Examples
        --------
        >>> from ITMO_FS.filters.multivariate import JMIM
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],[3, 1, 3, 1, 4],[4, 4, 3, 1, 5]],dtype = np.integer)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.integer)
        >>> model = JMIM(3)
        >>> model.fit(X, y)
        >>> model.selected_features_
        array([4, 0, 1])
    """

    def __init__(self, n_features, normalized=False):
        super().__init__()
        self.selected_features_ = None
        self.n_features = n_features
        self.normalized = normalized

    def __calc_jmi(self, fi, fs, y):
        if self.normalized:
            return symmetrical_relevance(fi, fs, y)
        else:
            return joint_mutual_information(fi, fs, y)

    def _fit(self, x, y):
        """
            Fits filter

            Parameters
            ----------
            x : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

            Returns
            -------
            None
        """

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" %
                (self.n_features, self.n_features_))

        features = generate_features(x)
        self.free_features = generate_features(x)

        mi = matrix_mutual_information(x, y)
        self.selected_scores = np.array([max(mi)], dtype='float')
        selected_index = np.argmax(mi)
        self.selected_features_ = np.array([selected_index], dtype='int')
        self.free_features = self.free_features[self.free_features !=
                                                selected_index]

        max_features = min(self.free_features.size, self.n_features - 1)
        for _ in range(max_features):
            jmi_scores = np.array(
                [min([self.__calc_jmi(x[:, fi], x[:, fs], y) for fs in
                      self.selected_features_])
                 for fi in self.free_features])
            best_jmi = jmi_scores.max()
            selected_index = self.free_features[np.argmax(jmi_scores)]
            self.selected_features_ = np.append(
                self.selected_features_, selected_index)
            self.free_features = self.free_features[self.free_features !=
                                                    selected_index]
            self.selected_scores = np.append(self.selected_scores, best_jmi)
