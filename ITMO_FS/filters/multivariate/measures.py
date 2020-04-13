import numpy as np

from ...utils.information_theory import calc_entropy
from ...utils.information_theory import calc_conditional_entropy
from ...utils.information_theory import mutual_information
from ...utils.information_theory import matrix_mutual_information
from ...utils.information_theory import calc_conditional_mutual_information
from ...utils.information_theory import calc_joint_mutual_information
from ...utils.information_theory import calc_interaction_information


def MIM(selected_features, free_features, X, y):
	return generalizedCriteria(selected_features, free_features, X, y, 0, 0)

def MRMR(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, 1/selected_features.size, 0)

def JMI(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, 1/selected_features.size, 1/selected_features.size)

def CIFE(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, 1, 1)

def MIFS(selected_features, free_features, X, y, beta):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, beta, 0)

def CMIM(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	vectorized_function = lambda free_feature : \
		min(np.vectorize(lambda selected_feature : \
			calc_conditional_mutual_information(X[:, free_feature], y, X[:, selected_feature]))(selected_features))
	return np.vectorize(vectorized_function)(free_features)

def ICAP(selected_features, free_features, X, y):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	relevance = matrix_mutual_information(X[:, free_features], y)
	redundancy = np.vectorize(lambda free_feature : np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(free_features)
	cond_dependancy = np.vectorize(lambda free_feature : \
		np.sum(np.apply_along_axis(calc_conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y)))(free_features)
	return relevance - np.maximum(redundancy - cond_dependancy, 0.) 

#Ask question what should happen if number of features user want is less than useful number of features
def generalizedCriteria(selected_features, free_features, X, y, beta, gamma):
	if(selected_features.size == 0):
		return matrix_mutual_information(X, y)
	relevance = np.apply_along_axis(mutual_information, 0, X[:, free_features], y)
	redundancy = np.vectorize(lambda free_feature : np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(free_features)
	cond_dependancy = np.vectorize(lambda free_feature : \
		np.sum(np.apply_along_axis(calc_conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y)))(free_features)
	return relevance - beta * redundancy + gamma * cond_dependancy
	
GLOB_MEASURE = {"MIM" : MIM,
				"MRMR" : MRMR,
				"JMI" : JMI,
				"CIFE" : CIFE,
				"MIFS" : MIFS,
				"CMIM" : CMIM,
				"ICAP" : ICAP,
				"generalizedCriteria": generalizedCriteria}



