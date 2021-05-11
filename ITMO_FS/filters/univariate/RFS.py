import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ...utils import l21_norm, matrix_norm, BaseTransformer


class RFS(BaseTransformer):
    """
    Performs the Robust Feature Selection via Joint L2,1-Norms Minimization
    algorithm.

        Parameters
        ----------
        n_features : int
            Number of features to select.
        gamma : float
            Regularization parameter.
        max_iterations : int
            Maximum amount of iterations to perform.
        epsilon : positive float
            Specifies the needed residual between the target functions from
            consecutive iterations. If the residual is smaller than epsilon,
            the algorithm is considered to have converged.

        Notes
        -----
        For more details see `this paper
        <https://papers.nips.cc/paper/3988-efficient-and-robust-feature-selection-via-joint-l21-norms-minimization.pdf/>`_.


        Examples
        --------
        >>> from ITMO_FS.filters.univariate import RFS
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 3, 1],[2, 2, 3, 3, 2], [1, 3, 3, 1, 3],\
[1, 1, 3, 1, 4],[2, 4, 3, 1, 5]])
        >>> y = np.array([1, 2, 1, 1, 2])
        >>> model = RFS(2).fit(X, y)
        >>> model.selected_features_
        array([0, 3], dtype=int64)
    """

    def __init__(self, n_features, gamma=1, max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def _fit(self, X, y):
        """
            Fits the algorithm.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples) or (n_samples, n_classes)
                The target values or their one-hot encoding.

            Returns
            ----------
            None
        """

        if self.epsilon < 0:
            raise ValueError(
                "Epsilon should be positive, %d passed" % self.epsilon)

        if self.n_features > self.n_features_:
            raise ValueError(
                "Cannot select %d features with n_features = %d" % (
                    self.n_features, self.n_features_))

        if len(y.shape) == 2:
            Y = y
        else:
            Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

        n_samples = X.shape[0]
        A = np.append(X, self.gamma * np.eye(n_samples), axis=1)
        D = np.eye(n_samples + self.n_features_)

        previous_target = 0
        for step in range(self.max_iterations):
            D_inv = np.linalg.inv(D)
            U = np.dot(np.dot(np.dot(D_inv, A.T),
                              np.linalg.inv(np.dot(np.dot(A, D_inv), A.T))), Y)
            diag = 2 * matrix_norm(U)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)

            target = l21_norm(U)
            if step > 0 and abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        rfs_score = matrix_norm(U[:self.n_features_])
        ranking = np.argsort(rfs_score)[::-1]
        self.selected_features_ = ranking[:self.n_features]
