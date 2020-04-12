import numpy as np

from ...utils.information_theory import __calc_entropy
from ...utils.information_theory import __calc_conditional_entropy
from ...utils.information_theory import __mutual_information
from ...utils.information_theory import __calc_conditional_mutual_information
from ...utils.information_theory import __calc_joint_mutual_information
from ...utils.information_theory import __calc_interaction_information


def calc_mutual_information(X, y):
	return np.apply_along_axis(__mutual_information, 0, X, y)

def MRMR(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return calc_mutual_information(X, y)
	relevance = np.apply_along_axis(__mutual_information, 0, X[:, free_features], y)
	redundancy = np.vectorize(lambda i: np.mean(calc_mutual_information(X[:, selected_features], X[:, i])))(free_features)
	return relevance - redundancy

def CMIM(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return calc_mutual_information(X, y)
	vectorized_function = lambda free_feature : min(np.vectorize(lambda selected_feature : __calc_conditional_mutual_information(X[:, free_feature], y, X[:, selected_feature]))(selected_features))
	return np.vectorize(vectorized_function)(free_features)

	
GLOB_MEASURE = {"MRMR" : MRMR,
				"CMIM" : CMIM}

# def FCBF(X, y):
#     freeXPool = list(range(0, X.shape[1]))
#     takenXPool = []
#     while(freeXPool == []):
#         max_inf = 0.0
#         max_index = 0
#         for i in range(X.shape[1]):
#             temp_inf = __mutual_information_single(X[:, i], y)
#             if temp_inf > max_inf:
#                 max_inf = temp_inf
#                 max_index = i
#         freeXPool.remove(max_index)
#         takenXPool.append(max_index)
#         poolCopy = freeXPool.copy()
#         for i in freeXPool:
#             relevance = mutual_information(X[:, i], y)
#             redundancy = mutual_information(X[:, i], X[:, max_index])
#             if redundancy > relevance:
#                 poolCopy.remove(i)
#         freeXPool = poolCopy
#     return takenXPool


