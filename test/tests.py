import pandas as pd

from filters.SpearmanCorrelation import SpearmanCorrelation
from wrappers import SequentialForwardSelection


def filter_test(filter, X, y, answer):
    try:
        assert filter.run(X, y).keys() == answer.keys()
        print('Test passed')
    except AssertionError:
        print('Test failed')


def wrapper_test(wrapper):
    pass  # TODO test wrapper algorithm


def sequential_forward_selection_test():
    wrapper_test(SequentialForwardSelection)
    pass  # TODO Implement this


def electricity_preprocess():
    data = pd.read_csv('data/electricity-normalized.csv')
    data['class'] = data['class'].apply(lambda x: 0 if x == 'DOWN' else 1)
    target = data['class']
    data.drop(['class'], axis=1)
    return data, target


electricity_X, electricity_y = electricity_preprocess()
filter_tests = [
    (SpearmanCorrelation(0.3), electricity_X, electricity_y,
     {3: 0.4282110555585918, 4: 0.42806820932633866, 6: 0.33097511862233125, 8: 1.0})
]

if __name__ == '__main__':
    for test in filter_tests:
        filter_test(*test)
