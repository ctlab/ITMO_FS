import pytest

from ITMO_FS.filters.multivariate import FCBFDiscreteFilter


pytestmark = pytest.mark.synthetic


def test_fcbf_discrete_filter_selects_requested_upper_bound(discrete_classification_data):
    x, y = discrete_classification_data

    filt = FCBFDiscreteFilter(delta=0.0).fit(x, y)
    transformed = filt.transform(x)

    assert transformed.shape[0] == x.shape[0]
    assert len(filt.selected_features_) > 0
    assert len(filt.selected_features_) <= x.shape[1]
