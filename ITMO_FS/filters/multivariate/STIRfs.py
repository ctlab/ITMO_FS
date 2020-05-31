import numpy as np
from sklearn.metrics import pairwise_distances

class STIRfs(object):
    def __init__(self, n):

        self.n = n
        self.feature_scores = None
        self.top_features = None

    def rang(self, X):
        #Computing denominator of diff formula
        #   for each attribute x (column) in X

        return (X.max(axis=0) - X.min(axis=0))

    def diff_func(self, a, b):
        #Compute the difference two vectors elementwise

        return np.abs(a - b)


    def get_distance(self, X):
        #Computing distance between two sample (instances)
        #based on all attributes, normalized by max-min

        max_min_vec = self.rang(X)
        min_vec = X.min(axis=0)

        X_centered = X - min_vec
        X_scale = X_centered / max_min_vec
        distance_X = pairwise_distances(X_scale)

        return distance_X


    def find_neighbors(self, X, y, k=1):
        #Find the nearest hit/miss matrices.

        dist_X = self.get_distance(X)
        num_samp = X.shape[0]

        idx = []
        hits = []
        misses = []

        for i in range(num_samp):
            distances = dist_X[i]
            nearest = np.argsort(distances)
            nearest_mat = np.column_stack((nearest, [y[j] for j in nearest]))
            nearest_hits = np.array(list(filter(lambda x: x[1] == nearest_mat[0][1],nearest_mat)))
            nearest_misses = np.array(list(filter(lambda x: x[1] != nearest_mat[0][1],nearest_mat)))

            hits += [r[0] for r in nearest_hits[1 : (k + 1)]]
            misses += [r[0] for r in nearest_misses[:k]]
            idx += [i for j in range(k)]

        hitmis_list = np.array([np.column_stack((idx, hits)),
                            np.column_stack((idx, misses))])

        return hitmiss_list

    def fit(self, X, y, k=1):
        #Computes the feature importance scores from the training data.

        neighbors_idx = self.find_neighbors(X, y, k)
        n_samp = X.shape[0]
        names_vec_w = X.shape[1]
        vec_w = np.zeros(names_vec_w)
        range_vec = np.array(self.rang(X))
        one_over_range = 1 / range_vec
        one_over_m = 1 / n_samples

        Ri_hit_idx, hit_idx = np.hsplit(neighbors_idx[0], 2)
        Ri_miss_idx, miss_idx = np.hsplit(neighbors_idx[1], 2)

        hit_idx = hit_idx.flatten()
        miss_idx = miss_idx.flatten()
        Ri_hit_idx = Ri_hit_idx.flatten()
        Ri_miss_idx = Ri_miss_idx.flatten()

        for attr_idx in range(names_vec_w):

            attr_val = X.T[attr_idx]
            hit_neighbors = np.array([attr_val[i] for i in hit_idx])
            miss_neighbors = np.array([attr_val[i] for i in miss_idx])
            hit_val = np.array([attr_val[i] for i in Ri_hit_idx])
            miss_val = np.array([attr_val[i] for i in Ri_miss_idx])

            attr_diff_hits = diff_func(hit_neighbors, hit_val) * one_over_range[attr_idx]
            attr_diff_misses = diff_func(miss_neighbors, miss_val) * one_over_range[attr_idx]
            mu_misses = np.sum(attr_diff_misses) / n_samp
            mu_hits = np.sum(attr_diff_hits) / n_samp

            vec_w[attr_idx] = mu_misses - mu_hits

        self.feature_score = vec_w * one_over_m
        self.top_features = np.argsort(self.feature_scores)[::-1]

    def transform(self, X):
        return X[:, self.top_features[:self.n]]

    def fit_transform(self, X, y, k):
        self.fit(X, y, k)
        return self.transform(X)