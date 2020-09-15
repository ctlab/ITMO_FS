import numpy as np
import pandas as pd

from ...utils import DataChecker, generate_features

class ModifiedTScore(DataChecker):
    """
        Performs the Modified T-score Feature Selection algorithm.

        Parameters
        ----------

        k : int
            Number of features to select.

        Notes
        -----
        For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.

        Examples
        --------

    """

    def __init__(self, p):
        self.k = k

    def fit(self, X, y, feature_names=None):
        '''
            Fits filter

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
              The training input samples
            y : numpy array, shape (n_samples, )
              The target values

            Returns
            ----------
            None

            See Also
            --------

            Examples
            --------
        '''

        features = generate_features(X)
        X, y, feature_names = self._check_input(X, y, feature_names)


        classes = np.unique(y)

        size_class0 = y[y == classes[0]].size
        size_class1 = y[y == classes[1]].size

        mean_class0 = np.mean(X[y == classes[0]], axis = 0)
        mean_class0 = np.nan_to_num(mean_class0)
        mean_class1 = np.mean(X[y == classes[1]], axis = 0)
        mean_class1 = np.nan_to_num(mean_class1)

        std_class0 = np.std(X[y == classes[0]], axis = 0)
        std_class0 = np.nan_to_num(std_class0)
        std_class1 = np.std(X[y == classes[1]], axis = 0)
        std_class1 = np.nan_to_num(std_class1)

        corr_with_y = np.apply_along_axis(lambda feature : abs(np.corrcoef(feature,y)[0][1]), 0, X)
        corr_with_y = np.nan_to_num(corr_with_y)

        corr_with_others = abs(pd.DataFrame(X).corr()).fillna(0).to_numpy()
        mean_of_corr_with_others = (corr_with_others.sum(axis = 1) - corr_with_others.diagonal())/corr_with_others.size

        t_score_numerator = abs(mean_class0 - mean_class1)
        t_score_denominator = np.sqrt((size_class0 * np.square(std_class0) + size_class1*np.square(std_class1))/(size_class0 + size_class1))
        modificator = corr_with_y/mean_of_corr_with_others
        modified_t_score = t_score_numerator / t_score_denominator * modificator

        features_indices = modified_t_score.argsort()[-self.k:][::-1]

        self.selected_features = features[features_indices]

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            ------
            Transformed 2D numpy array
        """

        if type(X) is np.ndarray:
            return X[:, self.selected_features]
        else:
            return X[self.selected_features]

    def fit_transform(self, X, y, feature_names=None):
        """
            Fits the filter and transforms given dataset X.

            Parameters
            ----------
            X : array-like, shape (n_features, n_samples)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.
            feature_names : list of strings, optional
                In case you want to define feature names

            Returns
            ------
            X dataset sliced with features selected by the filter
        """

        self.fit(X, y, feature_names)
        return self.transform(X)
