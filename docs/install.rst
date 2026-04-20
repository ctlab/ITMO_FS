########################
Install and contribution
########################

Prerequisites
=============

The feature selection library requires the following dependencies:

* python (>=3.10)
* numpy (>=1.25.2)
* scipy (>=1.11.4)
* scikit-learn (>=1.4.2)
* imbalanced-learn (>=0.14)
* qpsolvers (>=4)

Install
=======

ITMO_FS is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U ITMO_FS

If you prefer, you can clone the repository and install the package locally.
Use the following commands to get a copy from GitHub and install it::

  git clone https://github.com/ctlab/ITMO_FS.git
  cd ITMO_FS
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/ctlab/ITMO_FS.git

Development with uv
===================

For repository work, prefer `uv` dependency groups instead of exposing test
and documentation tooling as package extras::

  uv sync --group dev

Useful narrower scopes::

  uv sync --no-default-groups --group test
  uv sync --no-default-groups --group docs
  uv sync --only-group release

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

You can also use `pytest`::

  $ uv run --no-sync pytest -q

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/ctlab/ITMO_FS/pulls
