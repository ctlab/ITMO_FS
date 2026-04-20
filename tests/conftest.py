from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer


@pytest.fixture
def classification_data():
    x, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=42,
    )
    return x, y


@pytest.fixture
def discrete_classification_data(classification_data):
    x, y = classification_data
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    return discretizer.fit_transform(x), y


@pytest.fixture
def tiny_filter_example():
    x = np.array(
        [
            [3, 3, 3, 2, 2],
            [3, 3, 1, 2, 3],
            [1, 3, 5, 1, 1],
            [3, 1, 4, 3, 1],
            [3, 1, 2, 3, 1],
        ]
    )
    y = np.array([1, 3, 2, 1, 2])
    return x, y


@pytest.fixture
def madelon_data():
    dataset_path = Path(__file__).resolve().parent / "datasets" / "madelon.csv"
    if not dataset_path.exists():
        pytest.skip("tests/datasets/madelon.csv is not available locally")

    df = pd.read_csv(dataset_path, header=None)
    x = df.iloc[1:, :-1].to_numpy(dtype=float)
    y = df.iloc[1:, -1].to_numpy(dtype=int)
    return x, y
