from utils.data_check import *


class Melif:
    __classifiers = []
    __data = []
    __features = []

    def __init__(self, classifiers):
        self.__classifiers = classifiers

    def fit(self, data, features=None):
        if features is None:
            features = []
        check_data(data)
        check_features(features)
        self.__data = data
        self.__features = features


    def run(self):
        pass

    def coordinate_descend(self):
        pass
