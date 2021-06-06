from numpy import dot


def weight_fusion(filter_scores, weights):
    return filter_scores.T.dot(weights)
