from functools import partial, update_wrapper


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def select_best_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=True)


def select_worst_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=False)


def __select_by_value(scores, value, more=True):
    features = []
    for key, sc_value in scores.items():
        if more:
            if sc_value >= value:
                features.append(key)
        else:
            if sc_value <= value:
                features.append(key)
    return features


def select_k_best(k):
    return _wrapped_partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return _wrapped_partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if type(k) != int:
        raise TypeError("Number of features should be integer")
    if k > len(scores):
        raise ValueError("Cannot select %d features with n_features = %d" % (k, len(scores)))
    return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


def __select_percentage_best(scores, percent):
    features = []
    max_val = max(scores.values())
    threshold = max_val * percent
    for key, sc_value in scores.items():
        if sc_value >= threshold:
            features.append(key)
    return features


def select_best_percentage(percent):
    return _wrapped_partial(__select_percentage_best, percent=percent)


def __select_percentage_worst(scores, percent):
    features = []
    max_val = min(scores.values())
    threshold = max_val * percent
    for key, sc_value in scores.items():
        if sc_value >= threshold:
            features.append(key)
    return features


def select_worst_percentage(percent):
    return _wrapped_partial(__select_percentage_worst, percent=percent)


GLOB_CR = {"Best by value": select_best_by_value,
           "Worst by value": select_worst_by_value,
           "K best": select_k_best,
           "K worst": select_k_worst,
           "Worst by percentage": select_worst_percentage,
           "Best by percentage": select_best_percentage}
