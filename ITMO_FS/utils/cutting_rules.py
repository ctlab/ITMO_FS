import numpy as np
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
    if more:
        return np.flatnonzero(scores >= value)
    else:
        return np.flatnonzero(scores <= value)


def select_k_best(k):
    return _wrapped_partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return _wrapped_partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if not isinstance(k, int):
        raise TypeError("Number of features should be integer")
    if k > scores.shape[0]:
        raise ValueError(
            "Cannot select %d features with n_features = %d" % (k, len(scores))
        )
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]
    return order[:k]


def __select_percentage_best(scores, percent):
    return __select_k(scores, k=int(scores.shape[0] * percent), reverse=True)


def select_best_percentage(percent):
    return _wrapped_partial(__select_percentage_best, percent=percent)


def __select_percentage_worst(scores, percent):
    return __select_k(scores, k=int(scores.shape[0] * percent), reverse=False)


def select_worst_percentage(percent):
    return _wrapped_partial(__select_percentage_worst, percent=percent)


CR_NAMES = {
    "Best by value": select_best_by_value,
    "Worst by value": select_worst_by_value,
    "K best": select_k_best,
    "K worst": select_k_worst,
    "Worst by percentage": select_worst_percentage,
    "Best by percentage": select_best_percentage,
}
