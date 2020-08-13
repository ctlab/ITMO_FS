class DataChecker:

    @staticmethod
    def _check_input(X, y=None, feature_names=None):
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = list(range(X.shape[1]))
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            # TODO maybe something better instead of ravel() to fix the 2D array error?
            y = y.values.ravel()

        return X, y, feature_names

    def get_feature_names(self):
        return [name for (index, name) in self.feature_names.items() if index in self.selected_features]
