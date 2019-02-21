from numpy import array


def check_data(data):
    if type(data) is array:
        return
    if type(data) is list:
        return
    raise TypeError("input isn't a list ar numpy array")


def check_features(features):
    if all(isinstance(x, str) for x in features):
        return
    raise TypeError("Features should be strings")


def check_shapes(X, y):
    if X.shape[0] == y.shape[0]:
        return
    raise ValueError("Shape mismatch: {},{}".format(X.shape, y.shape))


def check_filters(filters):
    pass  # TODO check if current object has run


def check_classifier(classifier):
    pass  # TODO check if current object has fit and predict


def check_scorer(scorer):
    try:
        result = scorer(1, 5)
        if (result is float) or (result is int):
            raise AttributeError("Scorer isn't fit")
    except:
        pass  # todo scorer check


def check_cutting_rule(cutting_rule):
    pass # todo check cutting rule
