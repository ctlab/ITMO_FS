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

def check_classifier(classifier):
    pass #TODO check if current object has fit and predict