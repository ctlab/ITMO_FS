# ITMO_FS
Feature selection library in Python 

Package information: ![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)

Install with 

    pip install ITMO_FS
    
Current available algorithms:

| Filters                              | Wrappers                     | Hybrid |
|--------------------------------------|------------------------------|--------|
| Spearman correlation                 | Add Del                      | MeLiF  |
| Pearson correlation                  | Backward selection           |        |
| Fit Criterion                        | Sequential Forward Selection |        |
| F ratio                              |                              |        |
| Gini index                           |                              |        |
| Information Gain                     |                              |        |
| Minimum Redundancy Maximum Relevance |                              |        |
| VDM                                  |                              |        |

To use basic filter:
    
    from sklearn.datasets import load_iris
    from filters.Filter import * # provides you a filter class, basic measures and cutting rules
    
    data, target = load_iris(True)
    res = Filter("SpearmanCorr", GLOB_CR["Best by value"](0.9999)).run(data, target)
    print("SpearmanCorr:", data.shape, '--->', res.shape)

