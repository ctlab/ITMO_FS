import pytest

from ITMO_FS.filters.unsupervised import MCFS


pytestmark = pytest.mark.synthetic


def test_mcfs_selects_requested_number_of_features(classification_data):
    x, _ = classification_data

    filt = MCFS(n_features=3, k=2, p=3).fit(x)
    transformed = filt.transform(x)

    assert filt.selected_features_.shape == (3,)
    assert transformed.shape == (x.shape[0], 3)
