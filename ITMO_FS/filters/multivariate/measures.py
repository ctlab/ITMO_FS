import numpy as np

from ...utils.information_theory import calc_entropy
from ...utils.information_theory import calc_conditional_entropy
from ...utils.information_theory import mutual_information
from ...utils.information_theory import calc_conditional_mutual_information
from ...utils.information_theory import calc_joint_mutual_information
from ...utils.information_theory import calc_interaction_information


def calc_mutual_information(X, y):
	return np.apply_along_axis(mutual_information, 0, X, y)

def MRMR(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return calc_mutual_information(X, y)
	relevance = np.apply_along_axis(mutual_information, 0, X[:, free_features], y)
	redundancy = np.vectorize(lambda free_feature : np.mean(calc_mutual_information(X[:, selected_features], X[:, free_feature])))(free_features)
	return relevance - redundancy

def CMIM(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return calc_mutual_information(X, y)
	vectorized_function = lambda free_feature : \
		min(np.vectorize(lambda selected_feature : \
			calc_conditional_mutual_information(X[:, free_feature], y, X[:, selected_feature]))(selected_features))
	return np.vectorize(vectorized_function)(free_features)

	
GLOB_MEASURE = {"MRMR" : MRMR,
				"CMIM" : CMIM}



