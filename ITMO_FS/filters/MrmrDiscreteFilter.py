import random as rnd
from importlib import reload

import numpy as np
from sklearn.feature_selection import mutual_info_classif as MI
from sklearn.metrics import mutual_info_score as MI_features


class MrmrDiscreteFilter(object):
    """
        Creates mRMR (Minimum Redundancy Maximum Relevance) feature selection filter
        based on mutual information criteria for data with discrete features

        Parameters
        ----------
        number_of_features : int
            Amount of features to filter
        seed : int
            Seed for python random

        See Also
        --------
        http://home.penglab.com/papersall/docpdf/2005_TPAMI_FeaSel.pdf
        http://home.penglab.com/papersall/docpdf/2003_CSB_feasel.pdf


        examples
        --------
        # >>> mrmr_filter = MrmrDiscreteFilter(3)
        # >>> features = mrmr_filter.run(X, y)
        # >>> features
        ['v4938', 'v8837', 'v4520']

    """

    def __init__(self, number_of_features, seed=42):

        self.number_of_features = number_of_features
        rnd.seed = seed

    @staticmethod
    def _find_first_feature(X, y):

        max_mi = -1
        feature_index = 0

        for i in range(X.shape[1]):
            cur_mi = MI(X[:, i].reshape(-1, 1), y)
            if cur_mi > max_mi:
                feature_index = i
                max_mi = cur_mi

        return feature_index

    def _find_next_features(self, feature_set, not_used_features, X, y, info_gain):

        info_criteria = 0
        max_criteria = -1
        feature_index = 0

        for i in not_used_features:
            if info_gain == 'MID':
                info_criteria = self._MID(X[:, i], X[:, list(feature_set)], y)
            elif info_gain == 'MIQ':
                info_criteria = self._MIQ(X[:, i], X[:, list(feature_set)], y)
            if info_criteria > max_criteria:
                feature_index = i
                max_criteria = info_criteria

        return feature_index

    @staticmethod
    def _MID(A, B, y):
        return MI(A.reshape(-1, 1), y) - np.sum(
            [MI_features(A.ravel(), B[:, j].ravel()) for j in range(B.shape[1])]) / B.shape[1]

    @staticmethod
    def _MIQ(A, B, y):
        return MI(A.reshape(-1, 1), y) / (
                np.sum([MI_features(A.ravel(), B[:, j].ravel()) for j in range(B.shape[1])]) / B.shape[1])

    def run(self, X, y, info_gain='MID'):
        """
          Fits filter

          Parameters
          ----------
          X : numpy array or pandas DataFrame, shape (n_samples, n_features)
              The training input samples
          y : numpy array of pandas Series, shape (n_samples, )
              The target values
          info_gain : str, default 'MID'
              'MID' information criteria based on mutual information difference
              'MIQ' information criteria based on mutual information quotient

          Returns
          ----------
          used_features : list
              List of feature after mRMR filtering

          See Also
          --------

          examples
          --------

      """

        columns = None
        assert not 1 < X.shape[1] < self.number_of_features, 'incorrect number of features'

        return_feature_names = False

        try:
            import pandas

            if isinstance(X, pandas.DataFrame):
                return_feature_names = True
                columns = np.array(X.columns)
            else:
                pandas = reload(pandas)
        except ImportError:
            pass

        x = np.array(X)
        y = np.array(y).ravel()

        first_feature = self._find_first_feature(x, y)
        used_features = {first_feature}
        not_used_features = set([i for i in range(x.shape[1]) if i != first_feature])

        for _ in range(self.number_of_features - 1):
            feature = self._find_next_features(used_features, not_used_features, x, y, info_gain)
            used_features.add(feature)
            not_used_features.remove(feature)

        if return_feature_names:
            return list(columns[list(used_features)])

        return list(used_features)
