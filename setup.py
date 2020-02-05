import codecs

from setuptools import find_packages, setup

DISTNAME = 'ITMO_FS'
DESCRIPTION = 'Python Feature Selection library from ITMO University.'
with codecs.open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'N. Pilnenskiy'
MAINTAINER_EMAIL = 'somacruz@bk.ru'
URL = 'https://github.com/LastShekel/ITMO_FS'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/LastShekel/ITMO_FS'
VERSION = '0.2.1'
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'imblearn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
