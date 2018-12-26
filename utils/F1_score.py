def f1_score(actual, expected):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(actual)):
        if expected[i] == 1:
            if actual[i] == expected[i]:
                tp += 1
            else:
                fp += 1
        else:
            if actual[i] == expected[i]:
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
