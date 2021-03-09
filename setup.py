#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('rulecosi', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'rulecosi'
DESCRIPTION = 'A machine learning algorithm to combine and simplify rules ' \
              'from ensembles of decision trees. '
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'J. Obregon'
MAINTAINER_EMAIL = 'jobregon@khu.ac.kr'
URL = 'http://josue-obregon.com/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/jobregon1212/rulecosi'
VERSION = '0.0.1'
INSTALL_REQUIRES = ['pandas', 'numpy', 'scipy', 'scikit-learn',
                    'imbalanced-learn', 'bitarray']
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
    ],
    'xgboost':['xgboost'],
    'lightgbm':['lightgbm'],
    'catboost':['catboost'],
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
