from numpy import dot


def weight_fusion(filter_results, weights):
    result = {}
    for key, value in filter_results.items():
        result[key] = dot(value, weights)
    return result
