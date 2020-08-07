from numpy import array


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
    if features is None:
        try:
            if X.columns is list:
                features = X.columns
            else:
                features = list(X.columns)
        except AttributeError:
            features = [i for i in range(X.shape[1])]
    return array(features)


def check_shapes(X, y):
    if X.shape[0] == y.shape[0]:
        return
    raise ValueError("Shape mismatch: {},{}".format(X.shape, y.shape))


def check_filters(filters):
    for filter_ in filters:
        if not hasattr(filter_, 'fit'):
            raise TypeError(
                "filters should be a list of filters each implementing 'fit' method, %r was passed" % filter_)
        if not hasattr(filter_, 'transform'):
            raise TypeError(
                "filters should be a list of filters each implementing 'transform' method, %r was passed" % filter_)
        if not hasattr(filter_, 'fit_transform'):
            raise TypeError(
                "filters should be a list of filters each implementing 'fit_transform' method, %r was passed" % filter_)


def check_classifier(classifier):
    pass  # TODO check if current object has fit and predict


def check_scorer(scorer):
    try:
        result = scorer(1, 5)
        if (result is float) or (result is int):
            raise AttributeError("Scorer isn't fit")
    except AttributeError:
        pass  # todo scorer check


def check_cutting_rule(cutting_rule):
    pass  # todo check cutting rule


RESTRICTIONS = {'qpfs_filter': {'__select_k'}}


def check_restrictions(measure_name, cutting_rule_name):
    if measure_name in RESTRICTIONS.keys() and cutting_rule_name not in RESTRICTIONS[measure_name]:
        raise KeyError('This measure %r doesn\'t support this cutting rule %r' % (measure_name, cutting_rule_name))
