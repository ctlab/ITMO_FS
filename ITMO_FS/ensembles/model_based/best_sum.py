import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from logging import getLogger
from ...utils import BaseTransformer, apply_cr


class BestSum(BaseTransformer):
    """Best weighted sum ensemble. The ensemble fits the input models and
    computes the feature scores as the weighted sum of the models' feature
    scores and some performance metric (e.g. accuracy)

    Parameters
    ----------
    models : collection
        Collection of model objects. Models should have a fit(X, y) method and
        a field corresponding to feature weights.
    cutting_rule : string or callable
        A cutting rule name defined in GLOB_CR or a callable with signature
        cutting_rule (features), which should return a list features ranked by
        some rule.
    weight_func : callable
        The function to extract weights from the model.
    metric : string or callable
        A standard estimator metric (e.g. 'f1' or 'roc_auc') or a callable
        object / function with signature measure(estimator, X, y) which
        should return only a single value.
    cv : int
        Number of folds in cross-validation.

    See Also
    --------
    Jeon, H.; Oh, S. Hybrid-Recursive Feature Elimination for Efficient
    Feature Selection. Appl. Sci. 2020, 10, 3211.

        Examples
        --------
        >>> from ITMO_FS.ensembles import BestSum
        >>> from sklearn.svm import SVC
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.linear_model import RidgeClassifier
        >>> import numpy as np
        >>> models = [SVC(kernel='linear'),
        ... LogisticRegression(),
        ... RidgeClassifier()]
        >>> x = np.array([[3, 3, 3, 2, 2],
        ...               [3, 3, 1, 2, 3],
        ...               [1, 3, 5, 1, 1],
        ...               [3, 1, 4, 3, 1],
        ...               [3, 1, 2, 3, 1]])
        >>> y = np.array([1, 2, 2, 1, 2])
        >>> bs = BestSum(models, ("K best", 2),
        ... lambda model: np.square(model.coef_).sum(axis=0), cv=2).fit(x, y)
        >>> bs.selected_features_
        array([0, 2], dtype=int64)
    """

    def __init__(self, models, cutting_rule, weight_func, metric='f1_micro',
                 cv=3):
        super().__init__()
        self.models = models
        self.cutting_rule = cutting_rule
        self.weight_func = weight_func
        self.metric = metric
        self.cv = cv

    def _fit(self, X, y):
        """
            Fits the ensemble.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples, )
                The target values.

        Returns
        -------
        None
        """

        def __get_weights(model):
            _model = clone(model).fit(X, y)
            weights = self.weight_func(_model)
            perf = cross_val_score(_model, X, y, cv=self.cv,
                                   scoring=self.metric).mean()
            return weights * perf

        if len(self.models) == 0:
            getLogger(__name__).error("No models are set")
            raise ValueError("No models are set")

        model_scores = np.vectorize(
            lambda model: __get_weights(model),
            signature='()->(1)')(self.models)
        getLogger(__name__).info("Weighted model scores: %s", model_scores)
        self.feature_scores_ = model_scores.sum(axis=0)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        self.selected_features_ = apply_cr(self.cutting_rule)(
            self.feature_scores_)
