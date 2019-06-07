from numpy import array, random


def train_test_split(X, y, test_size):
    n_folds = int(1 / test_size)
    test_size = 1 - test_size
    t = list(zip(X, y))
    random.shuffle(t)
    X, y = array([i[0] for i in t]), array([i[1] for i in t])
    return X[:int(X.shape[0] * test_size)], y[:int(X.shape[0] * test_size)], \
           X[int(X.shape[0] * test_size):], y[int(X.shape[0] * test_size):]


def check_data(data):
    if type(data) is array:
        return
    if type(data) is list:
        return
    raise TypeError("input isn't a list ar numpy array")


def check_features(features, size):
    if all(isinstance(x, str) for x in features):
        if len(features) != size:
            raise IndexError("Shapes mismatch {} and {}".format(len(features), size))
        return
    else:
        raise TypeError("Features should be strings")


def generate_features(X, features=None):
    try:
        features = X.columns
    except AttributeError:
        if features is None:
            features = [i for i in range(X.shape[1])]
    return features


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
    pass  # todo check cutting rule
