from ITMO_FS.utils.qpfs_body import qpfs_body
from ITMO_FS.filters.measures import pearson_corr

def qpfs_filter(X, y, r = None, sigma = None, solv='quadprog', fn=pearson_corr):
	
	"""
    Performs Quadratic Programming Feature Selection algorithm.
    Note that this realization requires labels to start from 1 and be numberical.
    
    Parameters
    ----------
    X : array-like, shape (n_samples,n_features)
        The input samples.
    y : array-like, shape (n_samples)
        The classes for the samples.
    r : int
        The number of samples to be used in Nystrom optimization.
    sigma : double
        The threshold for eigenvalues to be used in solving QP optimization.
    solv : string, default
        The name of qp solver according to qpsolvers(https://pypi.org/project/qpsolvers/) naming.
        Note quadprog is used by default.
    fn : function(array, array), default
        The function to count correlation, for example pierson correlation or  mutual information.
        Note mutual information is used by default.
    Returns
    ------
    array-like, shape (n_features) : the ranks of features in dataset, with rank increase, feature relevance increases and redundancy decreases.
    
    See Also
    --------
	http://www.jmlr.org/papers/volume11/rodriguez-lujan10a/rodriguez-lujan10a.pdf
	
	Examples
	--------
	x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1], [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
	y = np.array([1, 3, 2, 1, 2])
	ranks = qpfs_filter(x, y)
	print(ranks)

    """

	return qpfs_body(X, y, r = r, sigma = sigma, solv = solv, fn = fn)
