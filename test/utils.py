import dvc.api
import pandas as pd
import pathlib
datasets = ["arcene.csv",
            "dexter.csv",
            "dorothea.csv",
            "gisette.csv",
            "madelon.csv"]


# def load_dataset(name):  # todo fails to hold header
#     with dvc.api.open(
#             'test/datasets/' + name) as fd:
#         df = pd.read_csv(fd, header=None)
#         features = ['v' + str(i) for i in range(df.shape[1] - 1)] + ["target"]
#         df.columns = features
#         return df
def load_dataset(name):
    return pd.read_csv(name)


def load_datasets():
    data = []
    for d in datasets:
        data.append(load_dataset(d))
    return data
