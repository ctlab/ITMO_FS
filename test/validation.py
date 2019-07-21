import re

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from filters import SpearmanCorrelationFilter, GiniIndexFilter, FitCriterion, InformationGainFilter, \
    SymmetricUncertainty, VDM


def breast_preprocess():
    data = pd.read_csv("data/matrix", sep=' ', header=None)
    target = data[data.columns[0]]
    data.drop(data.columns[-1], axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    return data, target


breast_X, breast_y = breast_preprocess()
breast = (breast_X, breast_y)
if __name__ == '__main__':
    GIF = GiniIndexFilter(1)
    IGF = InformationGainFilter(breast_X.shape[1])
    SPF = SpearmanCorrelationFilter(-1.0)
    FIF = FitCriterion()
    VDM = VDM()
    SU = SymmetricUncertainty.SymmetricUncertaintyFilter(breast_X.shape[1])

    python_res = SU.run(breast_X.values, breast_y.values)
    with open("C:\\Users\\SomaC\\IdeaProjects\\test_fs\\build\\resources\\main\\data\\result.txt", "r") as f:
        lines = []
        for i in f.readlines():
            lines.append((int(re.match("Key: (.*) Value: (.*)", i).group(1, 2)[0]),
                          float(re.match("Key: (.*) Value: (.*)", i).group(1, 2)[1])))
        java_res = dict(lines)
    s = 0
    java = [i[1] for i in list(sorted(java_res.items(), key=lambda x: x[0]))]
    python = [i[1] for i in list(sorted(SU.feature_scores.items(), key=lambda x: x[0]))]
    print(mean_absolute_error(python, java))
    print(mean_squared_error(python, java))
