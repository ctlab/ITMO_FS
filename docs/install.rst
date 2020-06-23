########################
Install and contribution
########################

Prerequisites
=============

The feature selection library requires the following dependencies:

* python (>=3.6)
* numpy (>=1.13.3)
* scipy (>=0.19.1)
* scikit-learn (>=0.22)
* imblearn (>=0.0)
* qpsolvers (>=1.0.1)

Install
=======

ITMO_FS is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U ITMO_FS

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/LastShekel/ITMO_FS.git
  cd ITMO_FS
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/LastShekel/ITMO_FS.git

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

You can also use `pytest`::

  $ pytest ITMO_FS -v

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/LastShekel/ITMO_FS/pulls
