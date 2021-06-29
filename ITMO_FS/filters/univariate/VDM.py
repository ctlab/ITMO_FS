import numpy as np

from ITMO_FS.utils.functions import cartesian
from ...utils import BaseTransformer


#  TODO some optimization and sklearn-like API
class VDM(BaseTransformer):
    """
        Creates Value Difference Metric builder.
        For continious features discretesation requered.

        Parameters
        ----------
        weighted: bool
            If weighted = False, modified version of metric which omits the
            weights is used
        q: int
            Power in VDM usually 1 or 2

        Notes
        -----
        For more details see papers about
        `Improved Heterogeneous Distance Functions
        <https://www.jair.org/index.php/jair/article/view/10182/>`_
        and `Implicit Future Selection with the VDM
        <https://aura.abdn.ac.uk/bitstream/handle/2164/10951/payne_ecai_98.pdf?sequence=1/>`_.

        Examples
        --------
        >>> x = np.array([[0, 0, 0, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 2]])
        >>> y = np.array([0,
        ...               1,
        ...               1])
        >>> vdm = VDM()
        >>> vdm.fit(x, y)
        array([[0.         4.35355339 4.        ]
               [4.5        0.         0.5       ]
               [4.         0.35355339 0.        ]])
    """

    def __init__(self, weighted=True, q=1):
        self.weighted = weighted
        self.q = q

    def _fit(self, X, y=None, **kwargs):
        """
            Generates metric for the data
            Complexity: O(n_features * n_samples^3) worst case, should be
            faster on a real data.

            Parameters
            ----------
            X: array-like, shape (n_features, n_samples)
                Input samples' parameters. Parameters among every class must be
                sequential integers.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential
                integers.
            Returns
            -------
            result:
                numpy.ndarray, shape=(n_samples, n_samples), dtype=np.double
                with selected version of metrics
            See Also
            --------
        """
        # TODO Fix case of y passed as DataFrame. For now y is transformed
        #  to 2D array and this causes an error. It seems better to follow
        #  usual sklearn practice and to use check_X_y but np.asarray(y[0])
        #  is also possible
        n_labels = np.max(y) + 1  # Number of different class labels
        n_samples = X.shape[0]  # Number of samples

        vdm = np.zeros((n_samples, n_samples),
                       dtype=np.double)  # Initializing output matrix

        for feature in X.T:  # For each attribute:
            # Initializing utility structures:
            n_values = np.max(
                feature) + 1  # Number of different values for the feature

            entries_x = np.empty(n_values,
                                 dtype=object)  # Array containing list of
            # indexes for every feature value
            entries_x[:] = [[] for _ in range(n_values)]

            entries_c_x = np.array(
                [{} for _ in range(n_labels)])  # Array of dirs of kind
            # {(feature value, amount of entries) for each class label

            for i, value in enumerate(feature):  # For each sample:
                entries_x[value].append(
                    i)  # Adding sample index to entries list
                entries_c_x[y[i]][value] = entries_c_x[y[i]].get(value,
                                                                 0) + 1  #
                # Adding entry for corresponding
                # class label

            amounts_x = np.array(list(
                map(len, entries_x)))  # Array containing amounts of samples
            # for every feature value

            # Calculating deltas:

            deltas = np.zeros((n_values, n_values),
                              dtype=np.double)  # Array for calculating deltas

            # Calculating components where exactly one of probabilities is
            # not zero:
            for c in range(n_labels):  # For each class:
                entries = np.array(list(entries_c_x[
                                            c].keys()))  # Feature values
                # which are presented in pairs for
                # the class
                amounts = np.array(
                    list(entries_c_x[c].values()))  # Corresponding amounts
                non_entries = np.arange(
                    n_values)  # Feature values which are not presented in pairs for the class
                # TODO get rid of error if entries are empty, example in test
                non_entries[entries] = -1
                non_entries = non_entries[non_entries != -1]

                for i in range(len(entries)):  # For each feature value
                    value = entries[i]  # Current value
                    v_c_instances = amounts[
                        i]  # Amount of instances with such value and such class
                    v_instances = amounts_x[
                        value]  # Amount of instances with such value
                    target_x, target_y = cartesian([value],
                                                   non_entries)  # Target indexes for deltas array
                    deltas[target_x, target_y] += (
                                                          v_c_instances / v_instances) ** 2
            deltas += deltas.T  # As we didn't determined indexes order, for each i, j some components are
            # written to delta(i, j) while others to delta(j, i), but exactly once. Adding transposed matrix to fix this

            # Calculating components where both probabilities are not zero:
            for c in range(n_labels):  # For each class:
                entries = np.array(list(entries_c_x[
                                            c].keys()))  # Feature values which are presented in pairs for
                # the class
                amounts = np.array(
                    list(entries_c_x[c].values()))  # Corresponding amounts
                probs = amounts / amounts_x[
                    entries]  # Conditional probabilities
                target_x, target_y = cartesian(np.arange(len(entries)),
                                               np.arange(len(
                                                   entries)))  # Target indexes
                # for deltas array
                deltas[entries[target_x], entries[target_y]] += (probs[
                                                                     target_x] -
                                                                 probs[
                                                                     target_y]) ** 2

            # Updating vdm:
            if not self.weighted:  # If non-weighted version of metrics was selected
                for i in range(n_values):  # For each value i
                    for j in range(n_values):  # For each value j
                        if amounts_x[i] == 0 or amounts_x[
                            j] == 0:  # If some value does not appear in current feature,
                            # skip it
                            continue
                        vdm[cartesian(entries_x[i], entries_x[j])] += \
                            deltas[i][j]
            else:  # If weighted version of metrics was selected
                weights = np.zeros(n_values,
                                   dtype=np.double)  # Initializing weights array
                for c in range(n_labels):  # For each class:
                    entries = np.array(list(entries_c_x[
                                                c].keys()))  # Feature values which are presented in pairs for
                    # the class
                    amounts = np.array(
                        list(entries_c_x[c].values()))  # Corresponding amounts
                    probs = amounts / amounts_x[
                        entries]  # Conditional probabilities
                    weights[entries] += probs ** 2
                weights = np.sqrt(weights)

                for i in range(n_values):  # For each value i
                    for j in range(n_values):  # For each value j
                        if amounts_x[i] == 0 or amounts_x[j] == 0:
                            continue
                        vdm[cartesian(entries_x[i], entries_x[j])] += \
                            deltas[i][j] * weights[i]

        return vdm
