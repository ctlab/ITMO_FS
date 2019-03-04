import numpy as np
from tqdm import tqdm

##TODO some optimization
class VDM:
    def __init__(self):
        pass

    def run(self, x, y, weighted=True):
        """
                Value Difference Metric
                http://aura.abdn.ac.uk/bitstream/handle/2164/10951/payne_ecai_98.pdf?sequence=1
                https://www.jair.org/index.php/jair/article/view/10182

                Parameters
                ----------
                x: array-like, shape (n_features, n_samples)
                    Input samples' parameters.
                y: array-like, shape (1, n_samples)
                    Input samples' class labels.
                weighted: bool
                    If weighted = False, modified version of metric used which omits the weights

                Returns
                -------
                result:
                    numpy.ndarray, shape (n_samples, n_samples) with selected version of metrics
                See Also
                --------
                Examples
                >>> x = np.array([[-2, 1, 1],
                ...               [3, 1, 2],
                ...               [3, 1, 1]])
                >>>
                >>> y = np.array([[7],
                ...               [3],
                ...               [3]])
                >>> VDM(x, y)
                array([[0.        , 2.70710678, 2.35355339],
                       [3.        , 0.        , 1.35355339],
                       [2.5       , 1.20710678, 0.        ]])
                --------
            """
        vdm = np.zeros((x.shape[0], x.shape[0]))  # Initializing output matrix

        for column in tqdm(x.T):  # For each attribute separately:
            # Initialising secondary structures:
            count_x_c = {}  # dict of dicts, for each attribute value contains distribution of class labels for samples
            count_x = {}  # dict, for each attribute value contains amount of samples with it
            for index, i in tqdm(enumerate(column)):  # For each sample
                count_x.setdefault(i, 0)
                count_x[i] += 1  # Increasing number of samples with attribute value == i

                c = y[index]  # Class label c of the corresponding sample
                count_x_c.setdefault(i, {})
                count_x_c[i].setdefault(c, 0)
                count_x_c[i][c] += 1  # Increasing number of samples with attribute value == i && class label == c

            # Calculating deltas:
            deltas = {}  # dict, For each pair of attribute values (i, j) contains delta(i, j)
            for i, i_count_c in tqdm(count_x_c.items()):  # For each attribute value i with its distribution i_count_c
                for j, j_count_c in tqdm(count_x_c.items()):  # For each attribute value j with its distribution j_count_c
                    delta = 0
                    count_i = count_x[i]  # Amount of samples with attribute value == i
                    count_j = count_x[j]  # Amount of samples with attribute value == j
                    for c, count_c in i_count_c.items():
                        delta += (count_c / count_i - j_count_c.get(c, 0) / count_j) ** 2

                    for c, count_c in j_count_c.items():
                        if c not in i_count_c:
                            delta += (count_c / count_x[j]) ** 2
                    deltas[(i, j)] = delta

            # Calculating weights if needed:
            if weighted:
                weights = {}  # dict, For each attribute value i contains weight(i)
                for i, i_count_c in count_x_c.items():  # For each attribute value i with its distribution i_count_c
                    weight = 0
                    for _, count_c in i_count_c.items():  # For each class label in the distribution of i
                        weight += count_c ** 2
                    weight = np.sqrt(weight)
                    weight /= count_x[i]
                    weights[i] = weight

            # Calculating VDM:
            if not weighted:
                for index_i, i in enumerate(column):
                    for index_j, j in enumerate(column):
                        vdm[index_i][index_j] += deltas[(i, j)]
            else:
                for index_i, i in enumerate(column):
                    for index_j, j in enumerate(column):
                        vdm[index_i][index_j] += deltas[(i, j)] * weights[i]
        return vdm
