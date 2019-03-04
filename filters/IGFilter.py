import numpy as np

class IGFilter:
    """
        Creates information gain feature selector
        TODO THEORY
        Parameters
        ----------
        n_features : int
            Number of features to select.

        See Xlso
        --------


        Examples
        --------
        >>> filter = IGFilter(4)
        >>> filter.run([1,2,3,2], np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0]]))
        [0, 3, 4, 1]
        """

    def __init__(self, n_features):
        self.n_features_ = n_features

    # TODO ADD METHOD DISCRIPTION
    def run(self, class_df_list, X):
        # Symmetric with array X.
        B = np.array([(sum(x) - x).tolist() for x in X])
        # Multidimensional array that does not appear in the classification.
        C = np.tile(class_df_list, (X.shape[0], 1)) - X
        N = sum(class_df_list)
        # Symmetrif with array C.
        D = N - X - B - C
        term_df_array = np.sum(X, axis=1)
        class_set_size = len(class_df_list)

        # Probability matrix when a feature exists.
        p_t = term_df_array / N
        # Probability matrix when a feature doesn't exists.
        p_not_t = 1 - p_t
        # Molecular plus one denominator plus two.
        p_c_t_mat = (X + 1) / (X + B + class_set_size)
        p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
        p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
        p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

        term_score_array = p_t * p_c_t + p_not_t * p_c_not_t

        sorted_term_score_index = term_score_array.argsort()[:: -1]
        # Return the list of indexes of the selected features.
        return list(sorted_term_score_index[:self.n_features_])
