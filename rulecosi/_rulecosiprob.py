"""RuleCOSI rule extractor.

This module contains the RuleCOSI extractor for classification problems.

The module structure is the following:

- The `BaseRuleCOSI` base class implements a common ``fit`` method
  for all the estimators in the module. This is done because in the future,
  the algorithm will work with regression problems as well.

- :class:`rulecosi.RuleCOSIClassifier` implements rule extraction from a
   variety of ensembles for classification problems.

"""

# Authors: Josue Obregon <jobregon@seoultech.ac.kr>
#
#
# License: MIT

import copy
import time
from abc import abstractmethod, ABCMeta
from ast import literal_eval
from math import sqrt
from operator import attrgetter

import numpy as np
import pandas as pd

import scipy.stats as st

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from rulecosi.rules import Rule, RuleSet
from .rule_extraction import RuleExtractorFactory
from .rule_heuristics import RuleHeuristics
from rulecosi._rulecosi import RuleCOSIClassifier, _ensemble_type, _statistical_error_estimate



class RuleCOSIProbClassifier(RuleCOSIClassifier):
    """ Tree ensemble Rule COmbiantion and SImplification algorithm for
       classification

       RuleCOSI extract, combines and simplify rules from a variety of tree
       ensembles and then constructs a single rule-based model that can be used
       for classification [1]. The ensemble is simpler and have a similar
       classification performance compared than that of the original ensemble.
       Currently only accept binary classification (March 2021)

       Parameters
       ----------
       base_ensemble: BaseEnsemble object, default = None
           A BaseEnsemble estimator object. The supported types are:
          - :class:`sklearn.ensemble.RandomForestClassifier`
          - :class:`sklearn.ensemble.BaggingClassifier`
          - :class:`sklearn.ensemble.GradientBoostingClassifier`
          - :class:`xgboost.XGBClassifier`
          - :class:`catboost.CatBoostClassifier`
          - :class:`lightgbm.LGBMClassifier`

          If the estimator is already fitted, then the parameters n_estimators
          and max_depth used for fitting the ensemble are used for the combination
          process. If the estimator is not fitted, then the estimator will be
          first fitted using the provided parameters in the RuleCOSI object.
          Default value is None, which uses a
          :class:`sklearn.ensemble.GradientBoostingClassifier` ensemble.

       n_estimators: int, default=5
           The number of estimators used for fitting the ensemble,
           if it is not already fitted.

       tree_max_depth: int, default=3
           The maximum depth of the individual tree estimators. The maximum
           depth limits the number of nodes in the tree. Tune this parameter
           for best performance; the best value depends on the interaction
           of the input variables.

       cov_threshold: float, default=0.0
           Coverage threshold of a rule to be considered for further combinations.
           (beta in the paper)The greater the value the more rules are discarded.
           Default value is 0.0, which it only discards rules with null coverage.

       conf_threshold: float, default=0.5
           Confidence or rule accuracy threshold of a rule to be considered for
           further combinations (alpha in the paper). The greater the value, the
           more rules are discarded. Rules with high confidence are accurate rules.
           Default value is 0.5, which represents rules with higher than random
           guessing accuracy.

       c= float, 0.25
           Confidence level for estimating the upper bound of the statistical
           correction of the rule error. It is used for the generalization process.

       percent_training= float, default=None
           Percentage of the training used for the combination and simplification
           process. If None, all the training data is usded. This is useful when
           the training data is too big because it helps to accelerate the
           simplification process. (experimental)

       early_stop: float, default=0
           This parameter allows the algorithm to stop if a certain amount of
           iterations have passed without improving the metric. The amount is
           obtained from the truncated integer of n_estimators * ealry_stop.

       metric: string, default='f1'
           Metric that is optimized in the combination process. The default is
           f1. Other accepted measures are:
            - 'roc_auc' for AUC under the ROC curve
            - 'accuracy' for Accuracy

       rule_order: string, default 'supp'
           Defines the way in the rules are ordered on each iteration. 'cov' order
           the rules first by coverage and 'conf' order the rules first by
           confidence or rule accuracy. 'supp' orders the rule by their support.
           This parameter affects the combination process and can be chosen
           conveniently depending on the desired results.

       sort_by_class: bool or list, default None
           Define if the combined rules are ordered by class. If 'None', the rules
           are not forced to be ordered (although in some cases the result will
           be ordered). If 'True', the rules are ordered by class
           lexicographically. If a list is provided, the rules will be ordered by
           class following the order provided by the user. The list should
           contain the actual classes passed in y when calling the fit method.

       column_names: array of string, default=None
           Array of strings with the name of the columns in the data. This is
           useful for displaying the name of the features in the generated rules.

       random_state: int, RandomState instance or None, default=None
           Controls the random seed given to the ensembles when trained. RuleCOSI
           does not have any random process, so it affects only the ensemble
           training.

       verbose: int, default=0
           Controls the output of the algorithm during the combination process. It
            can have the following values:
           - 0 is silent
           - 1 output only the main stages of the algorithm
           - 2 output detailed information for each iteration

       Attributes
       ----------
       X_ : ndarray, shape (n_samples, n_features)
           The input passed during :meth:`fit`.

       y_ : ndarray, shape (n_samples,)
           The labels passed during :meth:`fit`.

       classes_ : ndarray, shape (n_classes,)
           The classes seen at :meth:`fit`.

       original_rulesets_ : array of RuleSet, shape (n_estimators,)
           The original rulesets extracted from the base ensemble.

       simplified_ruleset_ : RuleSet
           Combined and simplified ruleset extracted from the base ensemble.

       n_combinations_ : int
           Number of rule-level combinations performed by the algorithm.

       combination_time_ : float
           Time spent for the combination and simplification process

       ensemble_training_time_ : float
           Time spent for the ensemble training. If the ensemble was already
           trained, this is 0.


       References
       ----------
       .. [1] Obregon, J., Kim, A., & Jung, J. Y., "RuleCOSI: Combination and
              simplification of production rules from boosted decision trees for
              imbalanced classification", 2019.

       Examples
       --------
       >>> from sklearn.ensemble import GradientBoostingClassifier
       >>> from sklearn.datasets import make_classification
       >>> from rulecosi import RuleCOSIClassifier
       >>> X, y = make_classification(n_samples=1000, n_features=4,
       ...                            n_informative=2, n_redundant=0,
       ...                            random_state=0, shuffle=False)
       >>> clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
       ...                          n_estimators=100, random_state=0)
       >>> clf.fit(X, y)
       RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
                          n_estimators=100, random_state=0)
       >>> clf.predict([[0, 0, 0, 0]])
       array([1])
       >>> clf.score(X, y)
       0.966...

       """

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 c=0.25,
                 percent_training=None,
                 early_stop=0,
                 metric='f1',
                 rule_order='supp',
                 sort_by_class=None,
                 float_threshold=1e-6,
                 column_names=None,
                 random_state=None,
                 verbose=0
                 ):
        super().__init__(base_ensemble=base_ensemble,
                         n_estimators=n_estimators,
                         tree_max_depth=tree_max_depth,
                         percent_training=percent_training,
                         early_stop=early_stop,
                         metric=metric,
                         float_threshold=float_threshold,
                         column_names=column_names,
                         random_state=random_state,
                         verbose=verbose
                         )

        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.c = c
        self.rule_order = rule_order
        self.sort_by_class = sort_by_class

    def fit(self, X, y):
        """ Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.



        Returns
        -------
        self : object
        """
        self._rule_extractor = None
        self._rule_heuristics = None
        self._base_ens_type = None
        self._weights = None
        self._global_condition_map = None
        self._bad_combinations = None
        self._good_combinations = None
        self._early_stop_cnt = 0
        self.alpha_half_ = None

        self.X_ = None
        self.y_ = None
        self.classes_ = None
        self.original_rulesets_ = None
        self.simplified_ruleset_ = None
        self.combination_time_ = None
        self.n_combinations_ = None
        self.ensemble_training_time_ = None

        # Check that X and y have correct shape
        if self.column_names is None:
            if isinstance(X, pd.DataFrame):
                self.column_names = X.columns
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.alpha_half_ = st.norm.ppf(1 - (self.c / 2))

        if self.percent_training is None:
            self.X_ = X
            self.y_ = y
        else:
            x, _, y, _ = train_test_split(X, y,
                                          test_size=(1 - self.percent_training),
                                          shuffle=True, stratify=y,
                                          random_state=self.random_state)
            self.X_ = x
            self.y_ = y

        # add rule ordering funcionality (2023.03.30)
        self._sorting_list = ['cov', 'conf', 'supp']
        self._sorting_list.remove(self.rule_order)
        self._sorting_list.insert(0, self.rule_order)
        self._sorting_list.reverse()
        if self.sort_by_class is not None:
            if isinstance(self.sort_by_class, bool):
                self.sort_by_class = self.classes_.tolist()

        if self.n_estimators is None or self.n_estimators < 2:
            raise ValueError(
                "Parameter n_estimators should be at "
                "least 2 for using the RuleCOSIProb method.")

        if self.verbose > 0:
            print('Validating original ensemble...')
        try:
            if self.base_ensemble is None:
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble_ = self.base_ensemble
            self._base_ens_type = _ensemble_type(self.base_ensemble_)
        except NotImplementedError:
            print(
                f'Base ensemble of type {type(self.base_ensemble_).__name__} '
                f'is not supported.')
        try:
            check_is_fitted(self.base_ensemble_)
            self.ensemble_training_time_ = 0
            if self.verbose > 0:
                print(
                    f'{type(self.base_ensemble_).__name__} already trained, '
                    f'ignoring n_estimators and '
                    f'tree_max_depth parameters.')
        except NotFittedError:
            self.base_ensemble_ = self._validate_and_create_base_ensemble()
            if self.verbose > 0:
                print(
                    f'Training {type(self.base_ensemble_).__name__} '
                    f'base ensemble...')
            start_time = time.time()
            self.base_ensemble_.fit(X, y)
            end_time = time.time()
            self.ensemble_training_time_ = end_time - start_time
            if self.verbose > 0:
                print(
                    f'Finish training {type(self.base_ensemble_).__name__} '
                    f'base ensemble'
                    f' in {self.ensemble_training_time_} seconds.')

        start_time = time.time()

        # First step is extract the rules
        self._rule_extractor = RuleExtractorFactory.get_rule_extractor(
            self.base_ensemble_, self.column_names,
            self.classes_, self.X_, self.y_, self.float_threshold)
        if self.verbose > 0:
            print(
                f'Extracting rules from {type(self.base_ensemble_).__name__} '
                f'base ensemble...')
        self.original_rulesets_, \
            self._global_condition_map = self._rule_extractor.extract_rules()
        processed_rulesets = copy.deepcopy(self.original_rulesets_)

        # We create the heuristics object which will compute all the
        # heuristics related measures
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=
                                               self._global_condition_map,
                                               cov_threshold=self.cov_threshold,
                                               conf_threshold=
                                               self.conf_threshold)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()

        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            for ruleset in processed_rulesets:
                for rule in ruleset:
                    new_A = self._remove_opposite_conditions(set(rule.A),
                                                             rule.class_index)
                    rule.A = new_A

        for ruleset in processed_rulesets:
            self._rule_heuristics.compute_rule_heuristics(
                ruleset, recompute=True)
        self.simplified_ruleset_ = processed_rulesets[0]

        self._simplify_rulesets(
            self.simplified_ruleset_)
        y_pred = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.compute_class_perf_fast(y_pred,
                                                         self.y_,
                                                         self.metric)
        self.simplified_ruleset_.rules.pop()

        self.n_combinations_ = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)

        if self.verbose > 0:
            print(f'Start combination process...')
            if self.verbose > 1:
                print(
                    f'Iteration {0}, Rule size: '
                    f'{len(self.simplified_ruleset_.rules)}, '
                    f'{self.metric}: '
                    f'{self.simplified_ruleset_.metric(self.metric)}')
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset_,
                                                    processed_rulesets[i])

            if self.verbose > 1:
                print(f'Iteration{i}:')
                print(
                    f'\tCombined rules size: {len(combined_rules.rules)} rules')
            # prune inaccurate rules
            self._sequential_covering_pruning(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSequential covering pruned rules size: '
                    f'{len(combined_rules.rules)} rules')
            # simplify rules
            self._simplify_rulesets(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSimplified rules size: '
                    f'{len(combined_rules.rules)} rules')

            # skip if the combined rules are empty
            if len(combined_rules.rules) == 0:
                if self.verbose > 1:
                    print(f'\tCombined rules are empty, skipping iteration.')
                continue
            self.simplified_ruleset_, best_ruleset = self._evaluate_combinations(
                self.simplified_ruleset_, combined_rules)

            if self._early_stop_cnt >= early_stop:
                break
            if self.simplified_ruleset_.metric() == 1:
                break

        self.simplified_ruleset_.rules[:] = [rule for rule in
                                             self.simplified_ruleset_.rules
                                             if rule.cov > 0]
        if self.verbose > 0:
            print(f'Finish combination process, adding default rule...')

        _ = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.prune_condition_map()
        end_time = time.time()
        self.combination_time_ = end_time - start_time
        if self.verbose > 0:
            print(
                f'R size: {len(self.simplified_ruleset_.rules)}, {self.metric}:'
                f' {self.simplified_ruleset_.metric(self.metric)}')

        return self