import numpy as np

from ITMO_FS.utils.information_theory import mutual_information, conditional_mutual_information


def _run(dataset, k):
    """
    :param dataset: inital dataset
    :param k: number of features used
    :return: index of the extracted informative feature
    """

    num, _ = dataset.shape
    matrix = np.zeros((k, k))
    r_cube = np.zeros((k, k, k))

    for i in range(k):
        for j in range(k):
            if i != j:
                matrix[i][j] = mutual_information(dataset[i], dataset[j])
    mutal_vector = [sum(matrix[i]) for i in range(k)]

    for k in range(k):
        for i in range(k):
            for j in range(k):
                if i != j and i != k and j != k:
                    r_cube[k][i][j] = matrix[i][j] - conditional_mutual_information(dataset[i], dataset[j],
                                                                                    dataset[k])
    r_matrix = [sum(r_cube[i]) for i in range(k)]
    r_vector = [sum(r_matrix[i]) for i in range(k)]

    for i in range(k):
        mutal_vector[i] -= r_vector[i]

    return np.argmax(mutal_vector)


class CIFE(object):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training instances to compute.

        Returns
        -------
        ans_list : list
        List of selected features.
        """
        copies_num, features_num = X.shape
        matrix = np.zeros((features_num - 1, copies_num))
        for i in range(features_num - 1):
            for j in range(copies_num):
                matrix[i][j] = X[j][i]
        ans_list = [0]
        for i in range(1, features_num):
            next_def = _run(matrix, i)
            if next_def not in ans_list:
                ans_list.append(next_def)
        return ans_list
