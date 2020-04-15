import numpy as np

from ...utils.information_theory import *

def MIM(selected_features, free_features, X, y):
	return matrix_mutual_information(X[:, free_features], y)

def MRMR(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, 1 / selected_features.size, 0)

def JMI(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return matrix_mutual_information(X, y)
	return generalizedCriteria(selected_features, free_features, X, y, 1 / selected_features.size, 1 / selected_features.size)

def CIFE(selected_features, free_features, X, y):
	return generalizedCriteria(selected_features, free_features, X, y, 1, 1)

def MIFS(selected_features, free_features, X, y, beta):
	return generalizedCriteria(selected_features, free_features, X, y, beta, 0)

def CMIM(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return matrix_mutual_information(X, y)
	vectorized_function = lambda free_feature : \
		min(np.vectorize(lambda selected_feature : \
			conditional_mutual_information(X[:, free_feature], y, X[:, selected_feature]))(selected_features))
	return np.vectorize(vectorized_function)(free_features)

def ICAP(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return matrix_mutual_information(X, y)
	relevance = matrix_mutual_information(X[:, free_features], y)
	redundancy = np.vectorize(lambda free_feature : np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(free_features)
	cond_dependency = np.vectorize(lambda free_feature : \
		np.sum(np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y)))(free_features)
	return relevance - np.maximum(redundancy - cond_dependency, 0.) 

def DCSF(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return np.vectorize(lambda x : 0)(free_features)
	vectorized_function = lambda free_feature : np.sum( \
		np.apply_along_axis(lambda z, x, y: conditional_mutual_information(x, y, z), 0, X[:, selected_features], X[:, free_feature], y) +
		np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], y, X[:, free_feature]) -
		matrix_mutual_information(X[:, selected_features], X[:, free_feature]))
	return np.vectorize(vectorized_function)(free_features)

def CFR(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return np.vectorize(lambda x : 0)(free_features)
	vectorized_function = lambda free_feature : np.sum( \
		np.apply_along_axis(lambda z, x, y: conditional_mutual_information(x, y, z), 0, X[:, selected_features], X[:, free_feature], y) +
		np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y) -
		matrix_mutual_information(X[:, selected_features], X[:, free_feature]))
	return np.vectorize(vectorized_function)(free_features)

def MRI(selected_features, free_features, X, y):
	return generalizedCriteria(selected_features, free_features, X, y, 2 / (selected_features.size + 1), 2 / (selected_features.size + 1))

def __information_weight(Xk, Xj, y):
	return 1 + (joint_mutual_information(Xk, Xj, y) + mutual_information(Xk, y) + mutual_information(Xj, y)) / (entropy(Xk) + entropy(Xj))

def __SU(Xk, Xj):
	return 2 * mutual_information(Xk, Xj) / (entropy(Xk) + entropy(Xj))

def IWFS(selected_features, free_features, X, y):
	if selected_features.size == 0:
		return np.vectorize(lambda x : 0)(free_features)
	vectorized_function = lambda free_feature : np.prod( \
		np.apply_along_axis(lambda Xj, Xk, y: __information_weight(Xk, Xj, y), 0, X[:, selected_features], X[:, free_feature], y) *
		(np.apply_along_axis(__SU, 0, X[:, selected_features], X[:, free_feature]) + 1))
	return np.vectorize(vectorized_function)(free_features)

#Ask question what should happen if number of features user want is less than useful number of features
def generalizedCriteria(selected_features, free_features, X, y, beta, gamma):
	if selected_features.size == 0:
		return matrix_mutual_information(X, y)
	relevance = np.apply_along_axis(mutual_information, 0, X[:, free_features], y)
	redundancy = np.vectorize(lambda free_feature : np.sum(matrix_mutual_information(X[:, selected_features], X[:, free_feature])))(free_features)
	cond_dependency = np.vectorize(lambda free_feature : \
		np.sum(np.apply_along_axis(conditional_mutual_information, 0, X[:, selected_features], X[:, free_feature], y)))(free_features)
	return relevance - beta * redundancy + gamma * cond_dependency
	
GLOB_MEASURE = {"MIM" : MIM,
				"MRMR" : MRMR,
				"JMI" : JMI,
				"CIFE" : CIFE,
				"MIFS" : MIFS,
				"CMIM" : CMIM,
				"ICAP" : ICAP,
				"DCSF" : DCSF,
				"CFR" : CFR,
				"MRI" : MRI,
				"IWFS" : IWFS,
				"generalizedCriteria": generalizedCriteria}



