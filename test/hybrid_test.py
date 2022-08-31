import unittest

from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import make_scorer
from ITMO_FS.filters import *
from ITMO_FS.wrappers import BackwardSelection
from ITMO_FS.utils import f1_scorer
from ITMO_FS.hybrid import FilterWrapperHybrid


class MyTestCase(unittest.TestCase):
    def test_est(self):
        classifier = LogisticRegression(max_iter=1000)
        back_selection = BackwardSelection(classifier, 2, make_scorer(f1_scorer))
        fw = FilterWrapperHybrid(
            UnivariateFilter(spearman_corr, cutting_rule=("K best", 2)), back_selection
        )
        check_estimator(fw)


if __name__ == "__main__":
    unittest.main()
