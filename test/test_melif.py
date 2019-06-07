import functools as ft
import warnings
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.svm import SVC

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
    estimator = SVC(kernel='poly')

    wrapper = Melif([SpearmanCorrelationFilter(cut), FitCriterion(),
                     SymmetricUncertainty.SymmetricUncertaintyFilter()], score=f1_score)

    with open("C:\\Users\\SomaC\\PycharmProjects\\features.txt", 'r') as f:
        feat = f.readlines()
    target = ['pd4']  # ["pd2", "pnp", "vp", "pd3", "msa", "de", "pd", "pd1","pd4"]
    for t in target:
        med_X, med_y = med_preprocess(t)
        wrapper.fit(med_X.values, med_y.values, points=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = []
        for i in range(5, 31):
            for delta in [0.5]:
                wr_cat = ft.partial(number_cut, number=i)
                f = wrapper.run(wr_cat, estimator, delta=delta)
                result.append((i, delta, wrapper.best_measure, wrapper.best_point, f))
        pd.DataFrame(result,
                     columns=['number of features from cut', 'delta', 'best score', 'best point',
                              'best features']).to_csv(
            t + '.csv', index=None)
