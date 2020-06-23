import random


def best_goes_first_fusion(filter_results, k):
    result = []
    place = 0
    filter_results = [[r[0] for r in result] for result in filter_results]
    while (len(result) < k) and (place < k):
        placed_features = list(map(list, zip(*filter_results)))[
            place]  # take only features at index=place in filter array
        random.shuffle(placed_features)
        [result.append(z) for z in list(set(placed_features)) if z not in result]
        place += 1
    return result[:k]


def borda_fusion(filter_results, k):
    filter_results = [{r[0]: p for r, p in zip(result, range(1, len(result) + 1))} for result in filter_results]
    result = filter_results[0]
    for filter_ in filter_results[1:]:
        for key in filter_.keys():
            result[key] += filter_[key]
    return list(dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)).keys())[:k]
