import numpy as np
import math


class SymmetricUncertaintyFilter:
    """
    __n_features: number of features with highest symmetric uncertainty value.


    """
    def __init__(self, n_features):
        self.__n_features = n_features

    @staticmethod
    def cal_entropy(y):
        dict_label = dict()
        for label in y[:, 0]:
            if label not in dict_label:
                dict_label.update({label: 1})
            else:
                dict_label[label] += 1
        entro = 0.0
        for i in dict_label.values():
            entro += -i / len(y[:, 0]) * math.log(i / len(y[:, 0]), 2)
        return entro

    def run(self, X, y):
        """
        X: shape(n_samples, n_features) feature matrix, an array filled with features.
        y: shape(n_samples, 1) label matrix, used to calculate SU value for each feature.

        :return: shape(n_samples, self.__n_features) an array filled with the selected features.
        """
        # Calculate the entropy of y.
        entropy = self.cal_entropy(y)

        # Calculate conditional entropy for each feature.
        list_f = list(dict())

        for index in range(len(X.T)):
            dict_i = dict()
            for i in range(len(X.T[index])):
                if X.T[index][i] not in dict_i:
                    dict_i.update({X.T[index][i]: [i]})
                else:
                    dict_i[X.T[index][i]].append(i)
            # print(dict_i)

            # Conditional entropy of a feature.
            con_entropy = 0.0
            # Entropy of each feature
            entropy_x = self.cal_entropy(X[:, [index]])
            # get corresponding values in y.
            for f in dict_i.values():
                # Probability of each class in a feature.
                p = len(f) / len(X.T[0])
                # Dictionary of corresponding probability in labels.
                dict_y = dict()
                for i in f:
                    if y.T[0][i] not in dict_y:
                        dict_y.update({y.T[0][i]: 1})
                    else:
                        dict_y[y.T[0][i]] += 1

                # calculate the probability of corresponding label.
                sub_entropy = 0.0
                for l in dict_y.values():
                    sub_entropy += -l / sum(dict_y.values()) * math.log(l / sum(dict_y.values()), 2)

                con_entropy += sub_entropy * p
            list_f.append({"su": 2 * (entropy - con_entropy) / (entropy_x + entropy), "index": index})

        # Sort by symmetric uncertainty in descending order.
        new_list = sorted(list_f, reverse=True, key=lambda k: k["su"])

        for item in new_list[self.__n_features:]:
            X = np.delete(X, [item.get("index")], axis=1)

        return X
