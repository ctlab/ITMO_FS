import numpy as np


class IGFilter:
    """
        Creates information gain feature selector

        Parameters
        ----------
        n_features : int
            Number of features to select.

        See Also
        --------


        Examples
        --------
        >>> filter = IGFilter(4)
        >>> filter.run([1,2,3,2], np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0]]))
        [0, 3, 4, 1]
        """

    def __init__(self, n_features):
        self.n_features_ = n_features

    def run(self, class_df_list, X):
        A = X
        # Symmetric with array A.
        B = np.array([(sum(x) - x).tolist() for x in A])
        # Multidimensional array that does not appear in the classification.
        C = np.tile(class_df_list, (A.shape[0], 1)) - A
        N = sum(class_df_list)
        # Symmetrif with array C.
        D = N - A - B - C
        term_df_array = np.sum(A, axis=1)
        class_set_size = len(class_df_list)

        # Probability matrix when a feature exists.
        p_t = term_df_array / N
        # Probability matrix when a feature doesn't exists.
        p_not_t = 1 - p_t
        # Molecular plus one denominator plus two.
        p_c_t_mat = (A + 1) / (A + B + class_set_size)
        p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
        p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
        p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

        term_score_array = p_t * p_c_t + p_not_t * p_c_not_t

        sorted_term_score_index = term_score_array.argsort()[:: -1]
        # Return the list of indexes of the selected features.
        return list(sorted_term_score_index[:self.n_features_])


