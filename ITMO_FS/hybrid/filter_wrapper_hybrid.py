class FilterWrapperHybrid:

    def __init__(self, filter_, wrapper):
        self._filter = filter_
        self._wrapper = wrapper
        self.selected_features = None
        self.best_score = None

    def fit(self, X, y):
        new = self._filter.fit_transform(X, y)
        self.selected_features = self._filter.selected_features
        self._wrapper.fit(new, y)
        self.best_score = self._wrapper.best_score

    def predict(self, X):
        new = X[:, self.selected_features]
        return self._wrapper.predict(new)
