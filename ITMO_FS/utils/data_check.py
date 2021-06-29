from numpy import array


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


def check_filters(filters):
    for filter_ in filters:
        attr = None
        if not hasattr(filter_, 'fit'):
            attr = 'fit'
        if not hasattr(filter_, 'transform'):
            attr = 'transform'
        if not hasattr(filter_, 'fit_transform'):
            attr = 'fit_transform'
        if not (attr is None):
            raise TypeError(
                "filters should be a list of filters each implementing {0} "
                "method, {1} was passed".format(attr, filter_))


def check_cutting_rule(cutting_rule):
    pass  # todo check cutting rule


RESTRICTIONS = {'qpfs_filter': {'__select_k'}}


def check_restrictions(measure_name, cutting_rule_name):
    if (measure_name in RESTRICTIONS.keys() and
            cutting_rule_name not in RESTRICTIONS[measure_name]):
        raise KeyError(
            "This measure %s doesn't support this cutting rule %s"
            % (measure_name, cutting_rule_name))
