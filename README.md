
<p align="center">
  <img src="docs/logos/logo_itmo_fs_itog_colour.jpg" width="420" alt="ITMO FS logo">
</p>

<h1 align="center">ITMO_FS</h1>

<p align="center">
  <strong>Feature selection library for Python</strong><br>
  Supervised, unsupervised, wrapper, embedded and hybrid methods under a unified interface.
</p>

<p align="center">
  <a href="https://pypi.org/project/ITMO-FS/">
    <img src="https://img.shields.io/pypi/v/ITMO-FS.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/ITMO-FS/">
    <img src="https://img.shields.io/pypi/pyversions/itmo-fs.svg" alt="Python versions">
  </a>
  <a href="https://github.com/ctlab/ITMO_FS/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/itmo-fs.svg" alt="License">
  </a>
  <a href="https://itmo-fs.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/itmo-fs/badge/?version=latest" alt="Documentation">
  </a>
</p>

---

## Overview

ITMO_FS is an open-source feature selection toolbox developed at ITMO University.

It implements dozens of classical and modern algorithms and exposes them via a scikit-learn-friendly API.

Typical use cases:

- Dimensionality reduction for high-dimensional datasets
- Preprocessing for traditional ML models (SVM, logistic regression, tree-based models, etc.)
- Exploratory analysis and feature ranking in research projects
- Benchmarking and comparison of feature selection algorithms

---

## Key features

- **Rich set of algorithms**: supervised / unsupervised filters, wrappers, embedded, hybrid and ensemble methods in one library.
- **Scikit-learn compatible API**: `fit`, `transform`, `fit_transform` and easy integration into `sklearn` pipelines.
- **Composable filters**: separate “measure” and “cutting rule” components let you implement custom strategies (e.g. thresholds or top-k).
- **Dense and sparse data support**: works with NumPy arrays, pandas DataFrames and SciPy sparse matrices.
- **Research background**: algorithms are based on well-known methods from the literature and used in research projects.

---

## Installation

From PyPI:

```bash
pip install -U ITMO_FS
```

From the latest source:

```bash
git clone https://github.com/ctlab/ITMO_FS.git
cd ITMO_FS
pip install .
```

---

## Quick start

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ITMO_FS.filters.univariate import UnivariateFilter, select_k_best
from ITMO_FS.filters.univariate import f_ratio_measure

# Synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=10,
    n_redundant=30,
    random_state=42,
)

# 1) Select 20 best features by F-ratio
fs = UnivariateFilter(
    measure=f_ratio_measure,
    cutting_rule=select_k_best(k=20),
)

# 2) Train a classifier on selected features
clf = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ("feature_selection", fs),
    ("classifier", clf),
])

pipe.fit(X, y)
print("Train accuracy:", pipe.score(X, y))
```

For more examples (wrappers, hybrids, ensembles), see the [documentation](https://itmo-fs.readthedocs.io/en/latest/).

---

## Algorithm families

ITMO_FS groups algorithms into several families:

- **Filters**
  - *Supervised*: correlation-based, information-theoretic, statistical, Relief-based, Laplacian and other measures.
  - *Unsupervised*: Laplacian / spectral scores, multi-cluster and discriminative feature selection.
- **Wrappers**: forward / backward selection, QPFS, hill-climbing, simulated annealing, recursive elimination.
- **Embedded methods**: MOSS, MOSNS, RFE and related model-based approaches.
- **Hybrid & ensembles**: MeLiF and other combinations of filters and wrappers.

---

## Full list of implemented algorithms

To keep this page readable, the detailed list of algorithms is hidden under a collapsible section:

<details>
<summary><strong>Click to expand the full list of algorithms</strong></summary>

### Supervised filters

- Spearman correlation  
- Pearson correlation  
- Fit Criterion  
- F ratio  
- Gini index  
- Symmetric Uncertainty  
- Fechner correlation  
- Kendall correlation  
- Information Gain  
- ANOVA  
- Chi-squared  
- Relief  
- ReliefF  
- Laplacian score  
- Modified T-score  
- Mutual Information Maximization  
- Minimum Redundancy Maximum Relevance  
- Joint Mutual Information  
- Conditional Infomax Feature Extraction  
- Mutual Information Feature Selection  
- Conditional Mutual Info Maximization  
- Interaction Capping  
- Dynamic Change of Selected Feature  
- Composition of Feature Relevancy  
- Max-Relevance and Max-Independence  
- Interaction Weight  
- Double Input Symmetric Relevance  
- Fast Correlation  
- Statistical Inference Relief  
- Trace Ratio (Fisher)  
- Nonnegative Discriminative Feature Selection  
- Robust Feature Selection  
- Spectral Feature Selection  
- VDM  
- QPFS  
- MIMAGA  

### Unsupervised filters

- Trace Ratio (Laplacian)  
- Multi-Cluster Feature Selection  
- Unsupervised Discriminative Feature Selection  

### Wrappers

- Add Del  
- Backward selection  
- Sequential Forward Selection  
- QPFS  
- Hill climbing  
- Simulated Annealing  
- Recursive Elimination  

### Hybrid methods

- Filter Wrapper  
- IWSSr-SFLA  

### Embedded methods

- MOSNS  
- MOSS  
- RFE  

### Ensembles

- MeLiF  
- Best goes first  
- Best sum  

</details>

---

## Documentation

Full documentation (tutorials and API reference) is available at:

- https://itmo-fs.readthedocs.io/en/latest/

---

## Contributing

Contributions are welcome!

- Report bugs and suggest features in the [issue tracker](https://github.com/ctlab/ITMO_FS/issues).
- Add new algorithms (filters, wrappers, embedded or hybrid methods).
- Improve documentation, examples and tests.

Please follow the existing code style and open a pull request when your change is ready.

---

## License

ITMO_FS is distributed under the BSD 3-Clause License.  
See the [`LICENSE`](LICENSE) file for details.