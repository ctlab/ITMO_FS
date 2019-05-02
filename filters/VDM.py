import numpy as np
from tqdm import tqdm

# TODO: Some more optimization
# TODO: Normalization with division by 2
class VDM:
    """
        Creates Value Difference Metric builder
        http://aura.abdn.ac.uk/bitstream/handle/2164/10951/payne_ecai_98.pdf?sequence=1
        https://www.jair.org/index.php/jair/article/view/10182

        Parameters
        ----------
        weighted: bool
            If weighted = False, modified version of metric which omits the weights is used

        See Also
        --------

        Examples
        --------
        >>> x = np.array([[0, 0, 0, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 2]])
        >>>
        >>> y = np.array([0,
        ...               1,
        ...               1])
        >>> vdm = VDM()
        >>> vdm.run(x, y)
        array([[0.         4.35355339 4.        ]
               [4.5        0.         0.5       ]
               [4.         0.35355339 0.        ]])
    """
    def __init__(self, weighted=True):
        self.weighted = weighted

    def run(self, x, y):
        """
            Generates metric for the data

            Parameters
            ----------
            x: array-like, shape (n_features, n_samples)
                Input samples' parameters. Parameters among every class must be sequential integers.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential integers.

            Returns
            -------
            result:
                numpy.ndarray, shape=(n_samples, n_samples), dtype=np.double with selected version of metrics
            See Also
            --------

        """
        x = np.asarray(x)  # Converting input data to numpy arrays
        y = np.asarray(y)

        n_labels = np.max(y) + 1  # Number of different class labels
        n_samples = x.shape[0]  # Number of samples

        vdm = np.zeros(n_samples * n_samples)  # Initializing output matrix, flatten shape used for better performance,
        # will be reshaped to (n_samples, n_samples) at the end

        for feature_index, feature in tqdm(enumerate(x.T)):  # For each attribute separately:
            # Initializing utility structures:
            count_x = {}  # Dict of kind {(attribute value, number of entries in column)}
            count_c_x = np.array([{} for _ in range(n_labels)])  # Array storing distribution of kind
            # {(attribute value, list of entries' indexes)} for each class label

            for i, value in tqdm(enumerate(feature)):  # For each sample
                count_x.setdefault(value, 0)  # Increasing number of samples with attribute value == feature[i]
                count_x[value] += 1
                count_c_x[y[i]].setdefault(value, [])  # Adding new entry for distribution with class label == y[i]
                count_c_x[y[i]][value].append(i)

            # Converting structures to numpy objects
            for distr in count_c_x:  # For each class label's distribution
                distr_arr = np.array(list(distr.items()))  # Converting distribution to numpy array
                for i in range(len(distr_arr)):  # Iterating over distribution with i
                    i_value, i_entries = distr_arr[i]  # (Attribute value,
                    # array of indexes with such value & class = current class)
                    c_i_total = len(i_entries)  # Amount of instances with class = i_value & class = current class
                    for j in range(i, len(distr_arr)):  # Iterating over distribution with j
                        j_value, j_entries = distr_arr[j]  # (Attribute value,
                        # array of indexes with such value & class = current class)
                        c_j_total = len(j_entries)  # Amount of instances with class = j_value & class = current class

                        i_total = count_x[i_value]  # Amount of instances with class = i_value
                        j_total = count_x[j_value]  # Amount of instances with class = j_value

                        # Calculating array of entries which are in both of arrays, only in first array,
                        # only in second array. Using two iterators technique according to the fact that both arrays
                        # are sorted by construction.
                        both = []
                        i_only = []
                        j_only = []
                        i_iter = 0
                        j_iter = 0
                        while i_iter < len(i_entries) and j_iter < len(j_entries):
                            i_elem = i_entries[i_iter]
                            j_elem = j_entries[j_iter]
                            if i_elem == j_elem:
                                both.append(i_elem)
                                i_iter = i_iter + 1
                                j_iter = j_iter + 1
                            elif i_elem < j_elem:
                                i_only.append(i_elem)
                                i_iter = i_iter + 1
                            else:
                                j_only.append(j_elem)
                                j_iter = j_iter + 1
                        i_only.extend(i_entries[i_iter:])
                        j_only.extend(j_entries[j_iter:])

                        # Converting indexes to flat-indexing form used in vdm array
                        both *= n_samples
                        both += feature_index
                        i_only *= n_samples
                        i_only += feature_index
                        j_only *= n_samples
















        dx = 5
            # Calculating deltas:
        #     for c in count_c_x:  # For each class label's distribution
        #
        #     deltas = np.empty((values_n, values_n))  # For each pair of attribute values (i, j) contains delta(i, j)
        #     for i in range(valuesN):  # For each value i
        #         for j in range(i, valuesN):  # For each value j
        #             delta = 0
        #             count_i = count_x[i]  # Amount of samples with attribute value == i
        #             count_j = count_x[j]  # Amount of samples with attribute value == j
        #             for c, count_c in count_x_c[i].items():  # Iterating over class tokens of i
        #                 delta += (count_c / count_i - count_x_c[j].get(c, 0) / count_j) ** 2
        #             for c, count_c in count_x_c[j].items():  # Iterating over class tokens of j
        #                 if c not in count_x_c[i]:  # Skipping tokens which have already been calculated
        #                     delta += (count_c / count_x[j]) ** 2
        #             deltas[i][j] = delta
        #             if i != j:
        #                 deltas[j][i] = delta
        #
        #     # Calculating VDM:
        #     if not self.weighted:
        #         for index_i, i in enumerate(column):
        #             for index_j, j in enumerate(column):
        #                 vdm[index_i][index_j] += deltas[i][j]
        #     else:  # if weighted metric was selected
        #         weights = np.empty(valuesN, dtype=np.double)  # For each attribute value i contains weight(i)
        #         for i in range(valuesN):  # For each attribute value i with its distribution i_count_c
        #             # Possible warning here: 'RuntimeWarning: invalid value encountered in double_scalars'
        #             # This may happen if some passed integers within single feature are not sequential
        #             weights[i] = np.sqrt(np.sum(np.array(list(count_x_c[i].values())) ** 2)) / count_x[i]
        #         for index_i, i in enumerate(column):
        #             for index_j, j in enumerate(column):
        #                 vdm[index_i][index_j] += deltas[i][j] * weights[i]
        # return vdm



class VDM_old:
    """
        Creates Value Difference Metric builder
        http://aura.abdn.ac.uk/bitstream/handle/2164/10951/payne_ecai_98.pdf?sequence=1
        https://www.jair.org/index.php/jair/article/view/10182

        Parameters
        ----------
        weighted: bool
            If weighted = False, modified version of metric which omits the weights is used

        See Also
        --------

        Examples
        --------
        >>> x = np.array([[0, 0, 0, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 2]])
        >>>
        >>> y = np.array([0,
        ...               1,
        ...               1])
        >>> vdm = VDM()
        >>> vdm.run(x, y)
        array([[0.         4.35355339 4.        ]
               [4.5        0.         0.5       ]
               [4.         0.35355339 0.        ]])
    """
    def __init__(self, weighted=True):
        self.weighted = weighted

    def run(self, x, y):
        """
            Generates metric for the data

            Parameters
            ----------
            x: array-like, shape (n_features, n_samples)
                Input samples' parameters. Parameters among every class must be sequential integers.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential integers.

            Returns
            -------
            result:
                numpy.ndarray, shape=(n_samples, n_samples), dtype=np.double with selected version of metrics
            See Also
            --------

        """
        x = np.asarray(x)  # Converting input data to numpy arrays
        y = np.asarray(y)

        vdm = np.zeros((x.shape[0], x.shape[0]))  # Initializing output matrix

        for column in tqdm(x.T):  # For each attribute separately:
            # Initializing utility structures:
            valuesN = np.max(column) + 1
            count_x = np.zeros(valuesN, dtype=np.int32)  # Array with amounts of samples for each attribute value
            count_x_c = np.array([{} for _ in range(valuesN)])  # Array of dicts with class labels distributions for each feature

            for index, value in tqdm(enumerate(column)):  # For each sample
                count_x[value] += 1  # Increasing number of samples with attribute value == i
                count_x_c[value].setdefault(y[index], 0)
                count_x_c[value][y[index]] += 1  # Increasing number of samples with attribute value == i && class label == y[index]

            # Calculating deltas:
            deltas = np.empty((valuesN, valuesN))  # For each pair of attribute values (i, j) contains delta(i, j)
            for i in range(valuesN):  # For each value i
                for j in range(i, valuesN):  # For each value j
                    delta = 0
                    count_i = count_x[i]  # Amount of samples with attribute value == i
                    count_j = count_x[j]  # Amount of samples with attribute value == j
                    for c, count_c in count_x_c[i].items():  # Iterating over class tokens of i
                        delta += (count_c / count_i - count_x_c[j].get(c, 0) / count_j) ** 2
                    for c, count_c in count_x_c[j].items():  # Iterating over class tokens of j
                        if c not in count_x_c[i]:  # Skipping tokens which have already been calculated
                            delta += (count_c / count_x[j]) ** 2
                    deltas[i][j] = delta
                    if i != j:
                        deltas[j][i] = delta

            # Calculating VDM:
            if not self.weighted:
                for index_i, i in enumerate(column):
                    for index_j, j in enumerate(column):
                        vdm[index_i][index_j] += deltas[i][j]
            else:  # if weighted metric was selected
                weights = np.empty(valuesN, dtype=np.double)  # For each attribute value i contains weight(i)
                for i in range(valuesN):  # For each attribute value i with its distribution i_count_c
                    # Possible warning here: 'RuntimeWarning: invalid value encountered in double_scalars'
                    # This may happen if some passed integers within single feature are not sequential
                    weights[i] = np.sqrt(np.sum(np.array(list(count_x_c[i].values())) ** 2)) / count_x[i]
                for index_i, i in enumerate(column):
                    for index_j, j in enumerate(column):
                        vdm[index_i][index_j] += deltas[i][j] * weights[i]
        return vdm
