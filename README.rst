.. -*- mode: rst -*-

|Travis|_

.. |Travis| image:: https://app.travis-ci.com/jobregon1212/rulecosi.svg?branch=master
.. _Travis: https://app.travis-ci.com/jobregon1212/rulecosi.svg?branch=master

RuleCOSI - Rule extraction COmbination and SImplification from classification tree ensembles
============================================================================================

.. _IAI: http://iai.khu.ac.kr/wiki/wiki.php
.. _Josue Obregon: https://josue-obregon.com/
.. _scikit-learn: http://scikit-learn.org/stable/

**RuleCOSI** is a machine learning package that combine and simplifies tree ensembles and generates
a single rule-based classifier that is smaller and simpler. It was developed in the Industrial Artificial
Intelligence Laboratory (`IAI`_) at Kyung Hee University by (`Josue Obregon`_). The implementation is compatible with scikit-learn_.

Installation
------------

Dependencies
~~~~~~~~~~~~

rulecosi is tested to work under Python 3.9+.
The dependency requirements used when developing the library are:

* numpy>=1.22.3
* scipy>=1.8.0
* scikit-learn>=1.0.2
* gmpy2>=2.1.2
* pandas>=1.4.1
* bitarray>=2.5.1
* xgboost>=1.5.2 (optional)
* lightgbm>=3.3.2 (optional)
* catboost>=1.0.4 (optional)

Installation
~~~~~~~~~~~~

From source available on GitHub
...............................

Right now it is just available from GitHub. You can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all basic dependencies::

  git clone https://github.com/jobregon1212/rulecosi.git
  cd rulecosi
  pip install .




This installs the basic rulecosi package. It will only work with the following scikit-learn tree ensembles:
BaggingClassifier, RandomForestClassifier and GradientBoostingClassifier.

If you want to install the package with support to other ensembles, you have to add the required packages separated
by commas inside square brackets when you install rulecosi. For example if you would like to have XGBoost support you
have to run the following command::

  git clone https://github.com/jobregon1212/rulecosi.git
  cd rulecosi
  pip install .[xgboost]

The supported optional packages are xgboost, lightgbm and catboost.

Documentation
-------------

The python documentation is available in `this link
<https://josue-obregon.com/rulecosi/>`_.

Development
-----------

The development of rulecosi tried to be in line with the one
of the scikit-learn community. Therefore, you can refer to their
`Development Guide
<http://scikit-learn.org/stable/developers>`_.

About
-----

If you use rulecosi in a scientific publication, we would appreciate
citations to the following paper::

    @article{obregon2019rulecosi,
      title={RuleCOSI: Combination and simplification of production rules from boosted decision trees for imbalanced classification},
      author={Obregon, Josue and Kim, Aekyung and Jung, Jae-Yoon},
      journal={Expert Systems with Applications},
      volume={126},
      pages={64--82},
      year={2019},
      publisher={Elsevier}
    }

The algorithm works with different type of ensembles and it uses the implementations provided by the sklearn package.
The supported tree ensemble types are:

    1. BaggingClassifier
    2. RandomForestClassifier
    3. GradientBoostingClassifier
    4. XGBClassifier
    5. LGBMClassifier
    6. CatBoostClassifier

For more information you can check the usage in the docstrings or the examples folder of this repository.


References:
-----------

.. [1] Obregon, J., Kim, A., & Jung, J. Y. (2019). RuleCOSI: Combination and simplification of production rules from boosted decision trees for imbalanced classification. Expert Systems with Applications, 126, 64-82.

