import pytest
from sklearn.linear_model import LogisticRegression

from ITMO_FS.wrappers.deterministic import SequentialForwardSelection


pytestmark = pytest.mark.synthetic


def test_sequential_forward_selection_smoke(classification_data):
    x, y = classification_data

    wrapper = SequentialForwardSelection(
        LogisticRegression(max_iter=1000),
        n_features=3,
        measure="f1_micro",
        cv=2,
    ).fit(x, y)

    transformed = wrapper.transform(x)

    assert wrapper.selected_features_.shape == (3,)
    assert transformed.shape == (x.shape[0], 3)
