#!/usr/bin/env python
import numpy as np

from sklearn.model_selection import cross_val_score

from ...utils import generate_features, BaseWrapper


class RecursiveElimination(BaseWrapper):
    """
        Performs a recursive feature elimination until the required number of
        features is reached.

        Parameters
        ----------
        estimator : object
            supervised learning estimator that should have a fit(X, y) method
            and a field corresponding to feature weights.
        n_features : int
            Number of features to leave.
        measure : string or callable
            A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable
            with signature measure(estimator, X, y) which should return only a
            single value.
        weight_func : callable
            The function to extract weights from the model.
        cv : int
            Number of folds in cross-validation.
            
        See Also
        --------
        Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., “Gene selection for
        cancer classification using support vector machines”, Mach. Learn.,
        46(1-3), 389–422, 2002.
        https://link.springer.com/article/10.1023/A:1012487302797

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from ITMO_FS.wrappers import RecursiveElimination
        >>> from sklearn.svm import SVC
        >>> import numpy as np
        >>> dataset = make_classification(n_samples=100, n_features=20,
        ... n_informative=4, n_redundant=0, shuffle=False, random_state=42)
        >>> x, y = np.array(dataset[0]), np.array(dataset[1])
        >>> model = SVC(kernel='linear')
        >>> rfe = RecursiveElimination(model, 5, measure='f1_macro',
        ... weight_func=lambda model: np.square(model.coef_).sum(axis=0)
        ... ).fit(x, y)
        >>> rfe.selected_features_
        array([ 0,  1,  2, 11, 19], dtype=int64)
    """

    def __init__(self, estimator, n_features, measure, weight_func, cv=3):
        self.estimator = estimator
        self.n_features = n_features
        self.measure = measure
        self.weight_func = weight_func
        self.cv = cv

    def _fit(self, X, y):
        """
            Fits the wrapper.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples)
                the target values.

            Returns
            ------
            None
        """

        if self.n_features > self.n_features_:
            raise ValueError("Cannot select %d features with n_features = %d" %
                (self.n_features, self.n_features_))

        self.selected_features_ = generate_features(X)

        while self.selected_features_.shape[0] != self.n_features:
            self._estimator.fit(X[:, self.selected_features_], y)
            least_important = np.argmin(self.weight_func(self._estimator))
            self.selected_features_ = np.delete(self.selected_features_,
                least_important)

        self.best_score_ = cross_val_score(self._estimator,
            X[:, self.selected_features_], y, cv=self.cv,
            scoring=self.measure).mean()
        self._estimator.fit(X[:, self.selected_features_], y)
