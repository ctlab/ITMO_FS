[//]: # ([![image]&#40;docs/logos/logo_itmo_fs_itog_colour.jpg&#41;]&#40;https://en.itmo.ru/&#41;)

<p align="center">
  <img src="docs/logos/logo_itmo_fs_itog_colour.jpg" width="500" alt="ITMO FS logo">
</p>

# ITMO\_FS

Feature selection library in Python

Package information:
![Python](https://img.shields.io/pypi/pyversions/itmo-fs.svg)
![License](https://img.shields.io/pypi/l/itmo-fs.svg)
![Docs](https://readthedocs.org/projects/itmo-fs/badge/?version=latest)

Install with

    pip install ITMO_FS

Current available algorithms:

| Supervised filters                           | Unsupervised filters                          | Wrappers                     | Hybrid         | Embedded | Ensembles       |
| -------------------------------------------- | --------------------------------------------- | ---------------------------- | -------------- | -------- | --------------- |
| Spearman correlation                         | Trace Ratio (Laplacian)                       | Add Del                      | Filter Wrapper | MOSNS    | MeLiF           |
| Pearson correlation                          | Multi-Cluster Feature Selection               | Backward selection           | IWSSr-SFLA     | MOSS     | Best goes first |
| Fit Criterion                                | Unsupervised Discriminative Feature Selection | Sequential Forward Selection |                | RFE      | Best sum        |
| F ratio                                      |                                               | QPFS                         |                |          |                 |
| Gini index                                   |                                               | Hill climbing                |                |          |                 |
| Symmetric Uncertainty                        |                                               | Simulated Annealing          |                |          |                 |
| Fechner correlation                          |                                               | Recursive Elimination        |                |          |                 |
| Kendall correlation                          |                                               |                              |                |          |                 |
| Information Gain                             |                                               |                              |                |          |                 |
| ANOVA                                        |                                               |                              |                |          |                 |
| Chi-squared                                  |                                               |                              |                |          |                 |
| Relief                                       |                                               |                              |                |          |                 |
| ReliefF                                      |                                               |                              |                |          |                 |
| Laplacian score                              |                                               |                              |                |          |                 |
| Modified T-score                             |                                               |                              |                |          |                 |
| Mutual Information Maximization              |                                               |                              |                |          |                 |
| Minimum Redundancy Maximum Relevance         |                                               |                              |                |          |                 |
| Joint Mutual Information                     |                                               |                              |                |          |                 |
| Conditional Infomax Feature Extraction       |                                               |                              |                |          |                 |
| Mutual Information Feature Selection         |                                               |                              |                |          |                 |
| Conditional Mutual Info Maximization         |                                               |                              |                |          |                 |
| Interaction Capping                          |                                               |                              |                |          |                 |
| Dynamic Change of Selected Feature           |                                               |                              |                |          |                 |
| Composition of Feature Relevancy             |                                               |                              |                |          |                 |
| Max-Relevance and Max-Independence           |                                               |                              |                |          |                 |
| Interaction Weight                           |                                               |                              |                |          |                 |
| Double Input Symmetric Relevance             |                                               |                              |                |          |                 |
| Fast Correlation                             |                                               |                              |                |          |                 |
| Statistical Inference Relief                 |                                               |                              |                |          |                 |
| Trace Ratio (Fisher)                         |                                               |                              |                |          |                 |
| Nonnegative Discriminative Feature Selection |                                               |                              |                |          |                 |
| Robust Feature Selection                     |                                               |                              |                |          |                 |
| Spectral Feature Selection                   |                                               |                              |                |          |                 |
| VDM                                          |                                               |                              |                |          |                 |
| QPFS                                         |                                               |                              |                |          |                 |
| MIMAGA                                       |                                               |                              |                |          |                 |

Documentation:

<https://itmo-fs.readthedocs.io/en/latest/>
