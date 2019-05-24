import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,LinearSVC
import functools as ft

from filters import FitCriterion, SymmetricUncertainty
from filters.SpearmanCorrelationFilter import SpearmanCorrelationFilter
from hybrid import Melif

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def cut(x, border=0.0):
    if type(x) is dict:
        return dict([i for i in x.items() if i[1] > border])
    if type(x) is np.ndarray or type(x) is list:
        return [i for i in x if i > border]


def number_cut(x, number=5):
    if type(x) is dict:
        return dict(list(sorted(x.items(), key=lambda x: x[1], reverse=True))[:number])
    # if type(x) is np.ndarray or type(x) is list:
    #     return [i for i in x if i > border]


def med_preprocess(feat):
    data = pd.read_csv("C:\\Users\\SomaC\\PycharmProjects\\" + feat + "pout.csv", sep=',', header=None)
    target = data[data.columns[-1]]
    data.drop(data.columns[-1], axis=1, inplace=True)
    return data, target


if __name__ == '__main__':
    estimator = LinearSVC()

    wrapper = Melif([SpearmanCorrelationFilter(), FitCriterion(),
                     SymmetricUncertainty.SymmetricUncertaintyFilter(0)], score=f1_score)

    with open("C:\\Users\\SomaC\\PycharmProjects\\features.txt", 'r') as f:
        feat = f.readlines()
    target = ["pd2", "pnp", "vp", "pd3", "msa", "de", "pd", "pd1", "pd4"]
    for t in target:
        med_X, med_y = med_preprocess(t)
        train_x, test_x, train_y, test_y = train_test_split(med_X, med_y, test_size=0.3)
        d = {}
        # for column in train_x.columns:
        #     estimator.fit(train_x[column].values.reshape((train_x.shape[0], 1)), train_y)
        #     predicted = estimator.predict(test_x[column].values.reshape((test_x.shape[0], 1)))
        #     d[column] = f1_score(test_y, predicted)
        #
        # pd.DataFrame(d.items(), index=None).to_csv(t + '_1.csv', index=None)

        wrapper.fit(med_X.values, med_y.values, points=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = []
        for i in range(2, 31):
            for delta in [0.2, 0.25, 0.5]:
                wr_cat = ft.partial(number_cut, number=i)
                f = wrapper.run(wr_cat, estimator, delta=delta)
                result.append((i, delta, wrapper.best_measure, wrapper.best_point, f))
        pd.DataFrame(result,
                     columns=['number of features from cut', 'delta', 'best score', 'best point',
                              'best features']).to_csv(
            t + '.csv', index=None)
