import numpy as np
from tqdm import tqdm

##TODO some more optimization
##TODO: ?? Normalization with division by 2
##TODO: ?? write converter
##TODO: ?? add types selection
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
                numpy.ndarray, shape (n_samples, n_samples) with selected version of metrics
            See Also
            --------
        """
        vdm = np.zeros((x.shape[0], x.shape[0]))  # Initializing output matrix

        for column in tqdm(x.T):  # For each attribute separately:
            # Initializing utility structures:
            valuesN = np.max(column) + 1
            count_x = np.zeros(valuesN, dtype=np.int32)  # Array with amounts of samples for each attribute value
            count_x_c = np.array([{} for _ in range(valuesN)])

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
                weights = np.empty(valuesN)  # For each attribute value i contains weight(i)
                for i in range(valuesN):  # For each attribute value i with its distribution i_count_c
                    weights[i] = np.sqrt(np.sum(np.array(list(count_x_c[i].values())) ** 2)) / count_x[i]
                for index_i, i in enumerate(column):
                    for index_j, j in enumerate(column):
                        vdm[index_i][index_j] += deltas[i][j] * weights[i]
        return vdm

