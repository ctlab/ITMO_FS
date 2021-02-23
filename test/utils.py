import dvc.api
import pandas as pd

datasets = ["arcene.csv",
            "dexter.csv",
            "dorothea.csv",
            "gisette.csv",
            "madelon.csv"]


def load_dataset(name):
    with dvc.api.open(
            'test/datasets/' + name) as fd:
        print(pd.read_csv(fd))


def load_datasets():
    data = []
    for d in datasets:
        data.append(load_dataset(d))
    return data
