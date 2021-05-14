import random
import numpy as np


def best_goes_first_fusion(filter_results, k):
    result = np.array([], dtype='int')
    place = 0
    while len(result) < k:
        placed_features = np.setdiff1d(filter_results[:, place], result)
        random.shuffle(placed_features)
        result = np.append(result, placed_features)
        place += 1
    return result[:k]


def borda_fusion(filter_results, k):
    n_features = filter_results.shape[1]
    scores = np.zeros(n_features)
    for f in filter_results:
        scores[f] += np.arange(1, n_features + 1)
    return np.argsort(scores)[:k]
