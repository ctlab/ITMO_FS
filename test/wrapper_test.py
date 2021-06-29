import unittest

import pandas as pd
from math import sqrt
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import make_scorer

from ITMO_FS import RecursiveElimination, BackwardSelection, AddDelWrapper, SequentialForwardSelection, \
    HillClimbingWrapper, SimulatedAnnealing, TPhMGWO
from ITMO_FS.utils.information_theory import *
from ITMO_FS.utils import test_scorer

np.random.seed(42)


class TestCases(unittest.TestCase):
    wide_classification = make_classification(n_features=2000, n_informative=100, n_redundant=500)
    tall_classification = make_classification(n_samples=50000, n_features=100, n_informative=23, n_redundant=30)
    wide_regression = make_regression(n_features=2000, n_informative=100)
    tall_regression = make_regression(n_samples=50000, n_features=200, n_informative=50)

    # def test_rec_elim(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     rec_elimination = RecursiveElimination(classifier, 10, 'f1')
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     rec_elimination.fit(X, y)
    #     features = rec_elimination.selected_features_
    #     assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score

    # def test_back_sel(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     back_selection = BackwardSelection(classifier, 10, 'f1')
    #     X, y = self.wide_classification
    #
    #     print('start calculating the default score')
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #     print('finish calculating the default score')
    #
    #     print('start backward selection')
    #     # TODO backward selection works for too long
    #     back_selection.fit(X, y)
    #     print('finish backward selection')
    #
    #     features = back_selection.selected_features_
    #     assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score

    # def test_add_del_wrapper(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     add_del_wrapper = AddDelWrapper(classifier, f1_score)
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     add_del_wrapper.fit(X, y)
    #     features = add_del_wrapper.selected_features_
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score
    #
    # def test_seq_forw_sel(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     seq_forw_sel = SequentialForwardSelection(classifier, 10, 'f1')
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     seq_forw_sel.fit(X, y)
    #     features = seq_forw_sel.selected_features_
    #     assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score

    # def test_qpfs_wrapper(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     seq_forw_sel = SequentialForwardSelection(LogisticRegression(), 10, 'f1')
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    #
    #     seq_forw_sel.fit(X, y)
    #     features = seq_forw_sel.selected_features
    #     assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1')
    #
    #     assert all(default_score < wrapper_score)

    # def test_hill_climbing(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     hill_climbing = HillClimbingWrapper(classifier, f1_score)
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     hill_climbing.fit(X, y)
    #     features = hill_climbing.selected_features_
    #     # assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score
    #
    # def test_sim_annealing(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     sim_annealing = SimulatedAnnealing(classifier, f1_score)
    #     X, y = self.wide_classification
    #
    #     sim_annealing.fit(X, y)
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     features = sim_annealing.selected_features_
    #     # assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score
    #
    # def test_wolves(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     tphmgwo = TPhMGWO()
    #     X, y = self.wide_classification
    #
    #     default_score = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    #
    #     tphmgwo.run(X, y)
    #     features = tphmgwo.selected_features_
    #     # assert len(features) == 10
    #
    #     wrapper_score = cross_val_score(classifier, X[:, features], y, cv=5, scoring='f1').mean()
    #
    #     assert default_score < wrapper_score
    #
    # def test_est(self):
    #     classifier = LogisticRegression(max_iter=1000)
    #     for f in [RecursiveElimination(classifier, 2, make_scorer(test_scorer)), BackwardSelection(classifier, 2, make_scorer(test_scorer)),
    #     AddDelWrapper(classifier, test_scorer), SequentialForwardSelection(classifier, 2, make_scorer(test_scorer)),
    #     HillClimbingWrapper(classifier, test_scorer), SimulatedAnnealing(classifier, test_scorer), TPhMGWO()]:
    #         check_estimator(f)

if __name__ == "__main__":
    unittest.main()
