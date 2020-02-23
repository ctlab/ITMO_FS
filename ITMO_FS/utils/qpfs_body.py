import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import mutual_info_score
from qpsolvers import solve_qp
import math    

def qpfs_body(X, y, fn, alpha = None, r = None, sigma = None, solv='quadprog'):
    #TODO understand why complex double appears 
    #TODO find suitable r parameter value
    #TODO find suitable sigma parameter value
	if r == None:
		r = X.shape[1] - 1
	if r >= X.shape[1]:
		raise TypeError("r parameter should be less than the number of features")
	F = np.zeros(X.shape[1], dtype=np.double) # F vector represents of each variable with class(here it is intialized)
	XT = X.T # Transposed matrix X 
	class_size = max(y) + 1# Count the number of classes, we assume that class labels would be numbers from 1 to max(y)
	priors = np.histogram(y, bins=max(y))[0] # Count prior probabilities of classes
	for i in range(1, class_size): # Loop through classes
		Ck = np.where(y == i, 1, 0) # Get array C(i) where C(k) is 1 when i = k and 0 otherwise
		F += priors[i - 1] * fn(XT, Ck) # Counting F vector
	Q = fn(XT, XT).reshape(XT.shape[0], XT.shape[0]) # Counting dependency, using normalized mutual info score
	indices = np.random.random_integers(0, Q.shape[0] - 1, r) # Taking random r indices according to Nystrom approximation
	A = Q[indices][:, :r] # A matrix for Nystrom(matrix of real numbers with size of [r, r])
	B = Q[indices][:, r:] # B matrix for Nystrom(matrix of real numbers with size of [r, M - r])
	if alpha == None:
		alpha = __countAlpha(A, B, F) # Only in filter method, in wrapper we should adapth it based on performance
	AInvSqrt = sqrtm(np.linalg.pinv(A)) # Calculate squared root of inverted matrix A
	S = np.add(A, AInvSqrt.dot(B).dot(B.T).dot(AInvSqrt)) # Caluclate S matrix
	eigvals, EVect = np.linalg.eig(S) # eigenvalues and eigenvectors of S
	U = np.append(A, B.T, axis = 0).dot(AInvSqrt).dot(EVect).dot(sqrtm(np.linalg.pinv(EVect))) # Eigenvectors of Q matrix using [A B]
	eigvalsFilt, UFilt = __filterBy(sigma, eigvals, U) # Take onyl eigenvalues greater than threshold and corresponding eigenvectors
	LFilt = np.zeros((len(eigvalsFilt), len(eigvalsFilt)), dtype = complex) # initialize diagonal matrix of eigenvalues
	for i in range(len(eigvalsFilt)): # Loop through eigenvalues
		LFilt[i][i] = eigvalsFilt[i] # Init diagonal values
	UFilt = np.array([list(map(lambda x: math.sqrt(x.imag**2 + x.real**2), t)) for t in UFilt])
	LFilt = np.array([list(map(lambda x: math.sqrt(x.imag**2 + x.real**2), t)) for t in LFilt])
	yf = solve_qp((1 - alpha) * LFilt, alpha * F.dot(UFilt), UFilt, np.zeros(UFilt.shape[0]), solver = solv) # perform qp on stated problem
	xSolution = UFilt.dot(yf) # Find x - weights of features
	forRanks = list(zip(xSolution, F, [x for x in range(len(F))])) # Zip into array of tuple for proper sort
	forRanks.sort(reverse = True)
	ranks = np.zeros(len(F)) 
	rankIndex = 1
	for i in forRanks:
		ranks[int(i[2])] = rankIndex
		rankIndex += 1
	return ranks


def __filterBy(sigma, eigvals, U):
	if sigma == None:
		return (eigvals, U)
	y = []
	for i in range(len(eigvals)):
		if eigvals[i] > sigma:
			y.append(i)
	return (eigvals[y], U[:, y])

def __countAlpha(A, B, F):
	sumQ = 0
	Comb = B.T.dot(np.linalg.pinv(A)).dot(B)
	sumQ = np.sum(A) + 2 * np.sum(B) + np.sum(Comb)
	sumQ /= (A.shape[1] + B.shape[1])**2
	sumF = np.sum(F)
	sumF /= len(F)
	return sumQ / (sumQ + sumF)
