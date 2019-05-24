from numpy import array, abs


def normalize(x):
    x = abs(array(x))
    max_ = max(x)
    min_ = max(x)
    return (x - min_) / (max_ - min_)
