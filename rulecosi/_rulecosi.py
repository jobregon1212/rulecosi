"""RuleCOSI rule extractor.

This module contains the RuleCOSI extractor for classification problems.

The module structure is the following:

- The `BaseRuleCOSI` base class implements a common ``fit`` method
  for all the estimators in the module. This is done because in the future,
  the algorithm will work with regression problems as well.

- :class:`rulecosi.RuleCOSIClassifier` implements rule extraction from a
   variety of ensembles for classification problems.

"""

# Authors: Josue Obregon <jobregon@khu.ac.kr>
#
#
# License: TBD

import copy
import time
from abc import abstractmethod, ABCMeta
from ast import literal_eval
from math import sqrt

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from .helpers import one_bitarray
from rulecosi.rules import Rule, RuleSet
from .rule_extraction import RuleExtractorFactory
from .rule_heuristics import RuleHeuristics


def _ensemble_type(ensemble):
    """ Return the ensemble type

    :param ensemble:
    :return:
    """
    if isinstance(ensemble, (BaggingClassifier, RandomForestClassifier)):
        return 'bagging'
    elif isinstance(ensemble, GradientBoostingClassifier):
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'xgboost.sklearn.XGBClassifier '
                                      'ensembles you should install xgboost '
                                      'library.')
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'lightgbm.sklearn.LGBMClassifier'>":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'lightgbm.sklearn.LGBMClassifier '
                                      'ensembles you should install lightgbm '
                                      'library.')
        return 'gbt'
    elif str(
            ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'catboost.core.CatBoostClassifier '
                                      'ensembles you should install catboost '
                                      'library.')
        return 'gbt'
    else:
        raise NotImplementedError


def _pessimistic_error_rate(N, e, z_alpha_half):
    """ Computes the statistical correction of the training error for
    estimating the generalization error.

    This function assumes that the errors on a rule (leaf node) follow a
    binomial distribution. Therefore, it computes the statistical correction
    as the upper limit of the normal approximation of a binomial distribution
    of the training error e.

    :param N: number of training records
    :param e: training error
    :param z_alpha_half: standardized value from a standard normal distribution
    :return: a float of the statistical correction of the training error e
    """
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(
        ((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N ** 2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator / denominator


class BaseRuleCOSI(BaseEstimator, metaclass=ABCMeta):
    """ Abstract base class for RuleCOSI estimators."""

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 m=0.5,
                 min_samples=1,
                 max_antecedents=5,
                 early_stop=0.30,
                 metric='f1',
                 column_names=None,
                 random_state=None,
                 verbose=0):

        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.m = m
        self.min_samples = min_samples
        self.max_antecedents = max_antecedents
        self.early_stop = early_stop
        self.metric = metric
        self.column_names = column_names
        self.random_state = random_state
        self.verbose = verbose

        self._rule_extractor = None
        self._rule_heuristics = None
        self._rule_simplifier = None
        self._base_ens_type = None
        self._weights = None
        self._global_condition_map = None
        self._bad_combinations = None
        self._good_combinations = None
        self._early_stop_cnt = 0

    def fit(self, X, y, sample_weight=None):
        """ Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        sample_weight: Currently this is not supported and it is here just
        for compatibility reasons

        Returns
        -------
        self : object
        """

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

        self.X_ = X
        self.y_ = y

        if self.n_estimators is None or self.n_estimators < 2:
            raise ValueError(
                "Parameter n_estimators should be at least 2 for using the RuleCOSI method.")

        if self.verbose > 0:
            print('Validating original ensemble...')
        try:
            self._base_ens_type = _ensemble_type(self.base_ensemble)
        except NotImplementedError:
            print(
                f'Base ensemble of type {type(self.base_ensemble).__name__} is not supported.')
        try:
            check_is_fitted(self.base_ensemble)
            self.ensemble_training_time_ = 0
            if self.verbose > 0:
                print(
                    f'{type(self.base_ensemble).__name__} already trained, ignoring n_estimators and '
                    f'tree_max_depth parameters.')
        except NotFittedError:
            self.base_ensemble = self._validate_and_create_base_ensemble()
            if self.verbose > 0:
                print(
                    f'Training {type(self.base_ensemble).__name__} base ensemble...')
            start_time = time.time()
            self.base_ensemble.fit(X, y, sample_weight=sample_weight)
            end_time = time.time()
            self.ensemble_training_time_ = end_time - start_time
            if self.verbose > 0:
                print(
                    f'Finish training {type(self.base_ensemble).__name__} base ensemble'
                    f' in {self.ensemble_training_time_} seconds.')
        start_time = time.time()

        # First step is extract the rules
        self._rule_extractor = RuleExtractorFactory.get_rule_extractor(
            self.base_ensemble, self.column_names,
            self.classes_, self.X_)
        if self.verbose > 0:
            print(
                f'Extracting rules from {type(self.base_ensemble).__name__} base ensemble...')
        self.original_rulesets_, self._global_condition_map = self._rule_extractor.extract_rules()
        processed_rulesets = copy.copy(self.original_rulesets_)
        # processed_rulesets = self.original_rulesets_

        # We create the heuristics object which will compute all the
        # heuristics related measures
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=self._global_condition_map,
                                               # cov_threshold=self.cov_threshold,
                                               conf_threshold=self.conf_threshold,
                                               min_samples=self.min_samples)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()
        self.simplified_ruleset_ = processed_rulesets[0]
        self._rule_heuristics.compute_rule_heuristics(self.simplified_ruleset_)
        # if str(self.base_ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
        self._simplify_rulesets(
            self.simplified_ruleset_)  ### change rulelat
        self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.compute_classification_performance(self.X_,
                                                                    self.y_)
        self.simplified_ruleset_.rules.pop()
        # else:
        #     self._simplify_rulesets(
        #         self.simplified_ruleset_)  ### change rulelat
        #     self.simplified_ruleset_.print_rules() #rulelat
        #     self.simplified_ruleset_.compute_classification_performance(self.X_, self.y_)

        self.n_combinations_ = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)
        # no_combinations = True  # flag to control if there are no combinations registered
        if self.verbose > 0:
            print(f'Start combination process...')
            if self.verbose > 1:
                print(
                    f'Iteration {0}, Rule size: {len(self.simplified_ruleset_.rules)}, '
                    f'{self.metric}: {self.simplified_ruleset_.metric(self.metric)}')
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            self._rule_heuristics.compute_rule_heuristics(processed_rulesets[i])
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
                    f'\tSequential covering pruned rules size: {len(combined_rules.rules)} rules')
            # simplify rules
            # combined_rules.print_rules()  # rulelat
            self._simplify_rulesets(combined_rules)
            # combined_rules.print_rules()  # rulelat
            if self.verbose > 1:
                print(
                    f'\tSimplified rules size: {len(combined_rules.rules)} rules')
            if self.verbose > 1:
                print(
                    f'\tCombined rules size: {len(combined_rules.rules)} rules')
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
            # if best_ruleset=='comb':
            #    no_combinations = False
            # print('#################################')
        # if no_combinations:  # if any combination was successful, we just simplify the first ruleset
        #     self._simplify_rulesets(self.simplified_ruleset_)

        self.simplified_ruleset_.rules[:] = [rule for rule in
                                             self.simplified_ruleset_.rules
                                             if rule.cov > 0]
        if self.verbose > 0:
            print(f'Finish combination process, adding default rule...')

        self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.prune_condition_map()
        end_time = time.time()
        self.combination_time_ = end_time - start_time
        if self.verbose > 0:
            print(
                f'R size: {len(self.simplified_ruleset_.rules)}, {self.metric}: '
                f'{self.simplified_ruleset_.metric(self.metric)}')

    def _validate_and_create_base_ensemble(self):
        """ Validate the parameter of base ensemble and if it is None, it set the default ensemble,
            GradientBoostingClassifier.

        """

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))
        if self.base_ensemble is None:
            if is_classifier(self):
                self.base_ensemble = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            elif is_regressor(self):
                self.base_ensemble = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                raise ValueError(
                    "You should choose an original classifier/regressor ensemble to use RuleCOSI method.")
        self.base_ensemble.n_estimators = self.n_estimators
        if str(
                self.base_ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
            self.base_ensemble.set_params(n_estimators=self.n_estimators,
                                          depth=self.tree_max_depth)
        elif isinstance(self.base_ensemble, BaggingClassifier):
            if is_classifier(self):
                self.base_ensemble.base_estimator = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble.base_estimator = DecisionTreeRegressor(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
        else:
            self.base_ensemble.max_depth = self.tree_max_depth
        return clone(self.base_ensemble)

    @abstractmethod
    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the
        combination and simplification process This includes the set of good
        combinations G and bad combinations B, but can include other sets
        necessary to the combination process
        """
        pass

    @abstractmethod
    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different
        criteria

        :param ruleset: ruleset R to which the default rule will be added
        """
        pass

    @abstractmethod
    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using
        the procedure described in the paper [ref]

        :param ruleset1: ruleset 1 to be combined

        :param ruleset2: ruleset 2 to be combined

        :return: a new ruleset containing the result of the combination process
        """
        pass

    @abstractmethod
    def _sequential_covering_pruning(self, ruleset):
        """ Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it
        sorts it and find the best rule. Then the covered instances are
        removed from the training set and the process is repeated until one
        of three stopping criteria are met: 1. all the records of the
        training set are covered, 2. all the rules on ruleset are used or 3.
        there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        pass

    @abstractmethod
    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset using the pessimist error.

        The function simplify the ruleset by iteratively removing conditions
        that minimize the pessimistic error. If all the conditions of a rule
        are removed, then the rule is discarded.

        :param ruleset:
        :return:
        """
        pass

    @abstractmethod
    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from
        each iteration cycle :param combined_rules: the combined rules that
        are obtained on each iteration of the combination cycle :return:the
        ruleset with best performance in accordance to the metric parameter
        """
        pass


class RuleCOSIClassifier(ClassifierMixin, BaseRuleCOSI):
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
        The greater the value the more rules are discarded. Default value is
        0.0, which it only discards rules with null coverage.

    conf_threshold: float, default=0.5
        Confidence or rule accuracy threshold of a rule to be considered for
        further combinations. The greater the value, the more rules are
        discarded. Rules with high confidence are accurate rules. Default value
        is 0.5, which represents rules with higher than random guessing
        accuracy.

    min_samples: int, default=1
        The minimum number of samples required to be at rule in the simplified
        ruleset.

    early_stop: float, default=0.30
        This parameter allows the algorithm to stop if a certain amount of
        iterations have passed without improving the metric. The amount is
        obtained from the truncated integer of n_estimators * ealry_stop.

    metric: string, default='gmean'
        Metric that is optimized in the combination process. The default is
        gmean because the algorithm was developed specially for imbalanced
        classification problems. Other accepted measures are:
         - 'f1' for F-measure
         - 'roc_auc' for AUC under the ROC curve
         - 'accuracy' for Accuracy

    column_names: array of string, default=None
        Array of strings with the name of the columns in the data. This is
        useful for displaying the name of the features in the generated rules.

    random_state: int, RandomState instance or None, default=None
        Controls the random seed given to the ensembles when trained. RuleCOSI
        does not have any random process, so it affects only the ensemble
        training.

    rule_order: string, default 'cov'
        Defines the way in the rules are ordered on each iteration. 'cov' order
        the rules first by coverage and 'conf' order the rules first by
        confidence or rule accuracy. This parameter affects the combination
        process and can be chosen conveniently depending the desired results.

    verbose: int, default=0
        Controls the output of the algorithm during the combination process. It
         can have the following values:
        - 0 is silent
        - 1 output only the main stages of the algorithm
        - 2 output information for each iteration

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
                 m=None,
                 min_samples=1,
                 max_antecedents=5,
                 early_stop=0.30,
                 metric='f1',
                 column_names=None,
                 random_state=None,
                 rule_order='conf',
                 verbose=0
                 ):
        super().__init__(base_ensemble=base_ensemble,
                         n_estimators=n_estimators,
                         tree_max_depth=tree_max_depth,
                         cov_threshold=cov_threshold,
                         m=m,
                         min_samples=min_samples,
                         max_antecedents=max_antecedents,
                         early_stop=early_stop,
                         metric=metric,
                         column_names=column_names,
                         random_state=random_state,
                         verbose=verbose
                         )

        self.conf_threshold = conf_threshold
        self.rule_order = rule_order

    def fit(self, X, y, sample_weight=None):
        """ Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        sample_weight: Currently this is not supported and it is here just for
        compatibility reasons

        Returns
        -------
        self : object
        """
        super().fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """ Predict classes for X.

        The predicted class of an input sample. The prediction use the
        simplified ruleset and evaluate the rules one by one. When a rule
        covers a sample, the head of the rule is returned as predicted class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class. The class with the highest value in the class
            distribution of the fired rule.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.simplified_ruleset_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is obtained from
        the class distribution of the fired rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        return self.simplified_ruleset_.predict_proba(X)

    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the
        combination and simplification process This includes the set of good
        combinations G and bad combinations B. It also includes the bitsets
        for the training data as well as the bitsets for each of the conditions
        """
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._rule_heuristics.initialize_sets()

    def _sort_ruleset(self, ruleset):
        """ Sort the ruleset in place according to the rule_order parameter.

        :param ruleset: ruleset to be ordered
        """
        if len(ruleset.rules) == 0:
            return
        if self.rule_order == 'cov':
            ruleset.rules.sort(key=lambda rule: (rule.cov, rule.conf, rule.supp,
                                                 -1 * len(rule.A),
                                                 rule.__str__()), reverse=True)

        elif self.rule_order == 'conf':
            ruleset.rules.sort(key=lambda rule: (rule.conf, rule.cov, rule.supp,
                                                 -1 * len(rule.A),
                                                 rule.__str__()), reverse=True)

    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using
        the procedure described in the paper [ref]

        Main guiding procedure for combining rulesets for classification,
        make a combination of each of the class with itself and all the other
        classes

        :param ruleset1: First ruleset to be combined

        :param ruleset2: Second ruleset to be combiuned

        :return: ruleset containing the combination of ruleset1 and ruleset 2
        """
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if
                              (rule1.y == [class_one])]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if
                              (rule2.y == [class_two])]
                combined_rules.update(
                    self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        combined_rules = RuleSet(list(combined_rules),
                                 self._global_condition_map)
        self._sort_ruleset(combined_rules)
        return combined_rules

    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        """ Actual combination procedure between to class-sliced rulesets

        :param s_ruleset1: sliced ruleset 1 according to a class

        :param s_ruleset2: sliced ruleset according to a class

        :return: a set of rules containing the combined rules of s_ruleset1 and
         s_ruleset2
        """
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A) == 0 or len(r2.A) == 0:
                    continue
                self.n_combinations_ += 1  # count the actual number of combinations
                # r1_AUr2_A = set({cond[0] for cond in r1.A}).union(
                #     {cond[0] for cond in r2.A})
                r1_AUr2_A = set(r1.A.union(r2.A))

                if frozenset(r1_AUr2_A) in self._bad_combinations:
                    continue

                heuristics_dict = self._rule_heuristics.get_conditions_heuristics(
                    r1_AUr2_A)
                if heuristics_dict['cov'] > 0:
                    samples_combination = np.array(
                        [heuristics_dict['cov_set'][i].count() for i in
                         range(len(self.classes_))]).sum()
                    cdp_data = np.array(
                        [heuristics_dict['cov_set'][i].count() for i in
                         range(len(self.classes_))]) / samples_combination
                else:
                    samples_combination = 0
                    cdp_data = np.array([0 for i in range(len(self.classes_))])

                    # create the new rule and compute class distribution and predicted class
                weight = None
                if self._base_ens_type == 'bagging':
                    if self._weights is None:
                        class_dist = np.mean([r1.class_dist, r2.class_dist],
                                             axis=0).reshape(
                            (len(self.classes_),))
                    else:
                        print('has weights')  # xgbchange
                        class_dist = np.average([r1.class_dist, r2.class_dist],
                                                axis=0,
                                                weights=[r1.weight,
                                                         r2.weight]).reshape(
                            (len(self.classes_),))
                        weight = (r1.weight() + r2.weight) / 2
                    y_class_index = np.argmax(class_dist).item()
                    y = np.array([self.classes_[y_class_index]])
                    logit_score = 0
                elif self._base_ens_type == 'gbt':
                    logit_score = r1.logit_score + r2.logit_score
                    # if len(self.classes_) == 2:
                    #     raw_to_proba = expit(logit_score)
                    #     class_dist = np.array(
                    #         [raw_to_proba.item(), 1 - raw_to_proba.item()])
                    # else:
                    #     class_dist = logit_score - logsumexp(logit_score)

                    class_dist = np.mean([r1.class_dist, r2.class_dist],
                                         axis=0).reshape(
                        (len(self.classes_),))  # xgbchang

                    # testing with smoothing the probability
                    # ========================================
                    # cdp_data = (r1.n_samples + r2.n_samples) / (
                    #         r1.n_samples + r2.n_samples).sum()

                    class_dist = np.average([class_dist, cdp_data],
                                            axis=0,
                                            weights=[self._weight_function(
                                                samples_combination),
                                                     1 - self._weight_function(
                                                         samples_combination)])

                    # ============================================

                    y_class_index = np.argmax(class_dist).item()
                    y = np.array([self.classes_[y_class_index]])
                elif self._base_ens_type == 'regressor':
                    y = np.mean([r1.y, r2.y], axis=0)

                if str(
                        self.base_ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
                    self._remove_opposite_conditions(r1_AUr2_A, y_class_index)

                # new_cond_set = {(cond, self._global_condition_map[cond])
                #                 for cond in r1_AUr2_A}
                # new_rule = Rule(frozenset(new_cond_set), class_dist=class_dist,
                new_rule = Rule(frozenset(r1_AUr2_A),
                                class_dist=class_dist,
                                logit_score=logit_score, y=y,
                                y_class_index=y_class_index,
                                classes=self.classes_, weight=weight)
                # new_rule = Rule(frozenset(r1_AUr2_A), class_dist=class_dist, logit_score=logit_score, y=y,
                #                 y_class_index=y_class_index, classes=self.classes_, weight=weight)

                # check if the combination was null before, if it was we just
                # skip it
                # if new_rule in self._bad_combinations:
                #     continue
                # if the combination was a good one before, we just add the
                # combination to the rules
                if new_rule in self._good_combinations:
                    heuristics_dict = self._good_combinations[new_rule]
                    new_rule.set_heuristics(heuristics_dict)
                    combined_rules.add(new_rule)
                else:
                    # heuristics_dict = self._rule_heuristics.get_conditions_heuristics(
                    #     r1_AUr2_A)
                    # new_cond_set)
                    new_rule.set_heuristics(heuristics_dict)
                    # new_rule_cov, new_rule_conf_supp = self.rule_heuristics.get_conditions_heuristics(r1_AUr2_A)
                    # new_rule.cov = new_rule_cov
                    # new_rule.conf = new_rule_conf_supp[y_class_index][0]
                    # new_rule.supp = new_rule_conf_supp[y_class_index][1]

                    # if new_rule.cov > self.cov_threshold and \
                    if new_rule.conf > self.conf_threshold:
                        combined_rules.add(new_rule)
                        self._good_combinations[new_rule] = heuristics_dict
                    else:
                        self._bad_combinations.add(frozenset(r1_AUr2_A))

        return combined_rules

    def _simplify_conditions(self, conditions):
        """ Remove redundant conditions of a single rule.

        Redundant conditions are conditions with the same attribute and same
        operator but different value.

        For example: (att1 > 5) ^ (att1 > 10). In this case we keep the
        second one because it contains the first one

        :param conditions: set of conditions' ids

        :return: return the set of conditions with no redundancy
        """
        cond_map = self._global_condition_map  # just for readability
        # create list with this format ["(att_index, 'OPERATOR')", 'cond_id']
        att_op_list = [
            [str((cond[1].att_index, cond[1].op.__name__)), cond[0]]
            for cond in conditions]
        att_op_list = np.array(att_op_list)
        # First part is to remove redundant conditions (e.g. att1>5 and att1> 10)
        # create list for removing redundant conditions
        dict_red_cond = {
            str(i[0]): att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 1]
            for i in att_op_list}
        # create generator to traverse just conditions with the same att_index and operator that appear more than once
        gen_red_cond = ((att_op, conds) for (att_op, conds) in
                        dict_red_cond.items() if len(conds) > 1)

        for (att_op, conds) in gen_red_cond:
            tup_att_op = literal_eval(att_op)
            list_conds = {cond_map[int(id_)] for id_ in conds}
            if tup_att_op[1] in ['lt', 'le']:
                edge_condition = max(list_conds, key=lambda
                    item: item.value)  # condition at the edge of the box
            if tup_att_op[1] in ['gt', 'ge']:
                edge_condition = min(list_conds, key=lambda item: item.value)
            list_conds.remove(
                edge_condition)  # remove the edge condition of the box from the list, so it will remain
            # [conditions.remove(hash(cond)) for cond in list_conds]
            [conditions.remove((hash(cond), cond)) for cond in list_conds]

        return frozenset(conditions)

    def _remove_opposite_conditions(self, conditions, class_index):
        """ Removes conditions that have disjoint regions and will make a
        rule to be discarded because it would have null coverage.

            This function is used with the trees generated with CatBoost
            algorithm, which are called oblivious trees. This trees share the
            same splits among entire levels. So The rules generated when
            combining tend to create many rules with null coverage, so this
            function helps to avoid this problem and explore better the
            covered feature space combination.

        :param conditions: set of conditions' ids

        :param class_index: predicted class of the rule created with the set of
        conditions

        :return: set of conditions with no opposite conditions
        """
        # att_op_list = [[self._global_condition_map[cond].att_index,
        #                 self._global_condition_map[cond].op.__name__, cond]
        #                for cond in conditions]
        att_op_list = [[cond[1].att_index,
                        cond[1].op.__name__, cond[0]]
                       for cond in conditions]
        att_op_list = np.array(att_op_list)
        # Second part is to remove opposite operator conditions (e.g. att1>=5  att1<5)
        # create list for removing opposite conditions
        dict_opp_cond = {
            str(i[0]): att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 2]
            for i in att_op_list}
        # create generator to traverse just conditions with the same att_index and different operator that appear
        # more than once
        gen_opp_cond = ((att, conds) for (att, conds) in dict_opp_cond.items()
                        if len(conds) > 1)
        for (_, conds) in gen_opp_cond:
            list_conds = [(int(id_),
                           self._rule_heuristics.get_conditions_heuristics(
                               # {int(id_)})['conf'][class_index])
                               {(int(id_),
                                 self._global_condition_map[int(id_)])})[
                               'conf'][class_index])
                          for id_
                          in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(
                best_condition)  # remove the edge condition of the box from the list so it will remain
            [conditions.remove((cond[0], self._global_condition_map[cond[0]]))
             for cond in list_conds]
        return frozenset(conditions)

    def _sequential_covering_pruning(self, ruleset):
        """Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it
        sorts it and find the best rule. Then the covered instances are
        removed from the training set and the process is repeated until one
        of three stopping criteria are met: 1. all the records of the
        training set are covered, 2. all the rules on ruleset are used or 3.
        there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        return_ruleset = []
        uncovered_instances = one_bitarray(self.X_.shape[0])
        found_rule = True
        while len(
                ruleset.rules) > 0 and uncovered_instances.count() > 0 and found_rule:
            self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                          uncovered_instances)
            self._sort_ruleset(ruleset)
            found_rule = False
            for rule in ruleset:
                if self._rule_heuristics.rule_is_accurate(rule,
                                                          uncovered_instances=uncovered_instances):
                    return_ruleset.append(rule)
                    ruleset.rules[:] = [rule for rule in ruleset if
                                        rule != return_ruleset[-1]]
                    found_rule = True
                    break
        ruleset.rules[:] = return_ruleset
        # self._sort_ruleset(ruleset) #rulelat

    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset inplace using the pessimist error.

        The function simplify the ruleset by iteratively removing conditions
        that minimize the pessimistic error. If all the conditions of a rule
        are removed, then the rule is discarded.

        :param ruleset:
        """
        for rule in ruleset:
            rule.A = self._simplify_conditions(set(rule.A))
            # RuleSet([rule], self._global_condition_map).print_rules() #rulelat
            base_line_error = self._compute_pessimistic_error(rule.A,
                                                              rule.class_index)
            min_error = 0
            while min_error <= base_line_error and len(rule.A) > 0:
                # errors = [(cond, self._compute_pessimistic_error(rule.A.difference([cond]), rule.class_index))
                #           for cond in rule.A]
                errors = [(cond,
                           self._compute_pessimistic_error(
                               rule.A.difference({cond}), rule.class_index),
                           self._rule_heuristics.get_conditions_heuristics(
                               [cond]),
                           str(cond[1])) for cond
                          in rule.A]

                # print([(err[0][1], err[1],
                #         err[2]['cov'],
                #         err[2]['conf'][rule.class_index],
                #         err[2]['supp'][rule.class_index],
                #         err[3]) for err
                #        in sorted(errors, key=lambda tup: tup[3])])  # rulelat
                # min_error_tup = min(errors, key=lambda tup: tup[1])
                min_error_tup = min(errors, key=lambda tup: (tup[1],
                                                             tup[2]['cov'],
                                                             tup[2]['conf'][
                                                                 rule.class_index],
                                                             tup[2]['supp'][
                                                                 rule.class_index],
                                                             tup[3]))
                # print('min: ', min_error_tup[0][1],
                #       min_error_tup[1])
                min_error = min_error_tup[1]
                if min_error <= base_line_error:
                    base_line_error = min_error
                    min_error = 0
                    rule_conds = set(rule.A)
                    rule_conds.remove(min_error_tup[0])
                    rule.A = frozenset(rule_conds)

        # min_cov = self.min_samples / self.X_.shape[0]
        ruleset.rules[:] = [rule for rule in ruleset
                            if 0 < len(rule.A)  # <= self.max_antecedents
                            and rule.cov > self.cov_threshold
                            and rule.conf > self.conf_threshold]

        # self._rule_heuristics.compute_rule_heuristics(ruleset, sequential_coverage=False)
        self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                      sequential_covering=True)
        self._sort_ruleset(ruleset)
        # self.rule_heuristics.compute_rule_heuristics(ruleset, sequential_coverage=True)

    def _compute_pessimistic_error(self, conditions, class_index):
        """ Computes a statistical correction to the training error to
        estimate the generalization error of one rule.

        This function assumes that the errors on the rule follow a binomial
        distribution. Therefore, it computes the statistical correction as
        the upper limit of the normal approximation of a binomial
        distribution of the training error e.

        :param conditions: set of conditions' ids

        :param class_index: predicted class index of the rule

        :return: the statistical correction of the training error of that
        rule (between 0 and 100)
        """
        if len(conditions) == 0:
            e = (self.X_.shape[0] - self._rule_heuristics.training_bit_sets[
                class_index].count()) / self.X_.shape[0]
            return 100 * _pessimistic_error_rate(self.X_.shape[0], e, 1.15)
        # cov, class_cov = self.rule_heuristics.get_conditions_heuristics(conditions, return_set_size=True)
        heuristics_dict = self._rule_heuristics.get_conditions_heuristics(
            conditions)
        total_instances = heuristics_dict['cov_set'][-1].count()
        accurate_instances = heuristics_dict['cov_set'][class_index].count()

        error_instances = total_instances - accurate_instances
        alpha_half = 1.15  # 25 % confidence for C4.5
        e = error_instances / total_instances  # totalInstances

        return 100 * _pessimistic_error_rate(total_instances, e, alpha_half)

    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different
        criteria

        :param ruleset: ruleset R to which the default rule will be added
        """
        if len(ruleset.rules) > 0:
            uncovered_instances = ~ruleset._predict(self.X_)[1]
        else:
            uncovered_instances = np.ones((self.X_.shape[0],),
                                          dtype=bool)

        all_covered = False
        if uncovered_instances.sum() == 0:
            uncovered_dist = np.array(
                [self._rule_heuristics.training_bit_sets[i].count() for i in
                 range(len(self.classes_))])
            all_covered = True
        else:
            uncovered_labels = self.y_[uncovered_instances]
            uncovered_dist = np.array(
                [(uncovered_labels == class_).sum() for class_ in
                 self.classes_])

        default_class_idx = np.argmax(uncovered_dist)
        default_rule = Rule({},
                            class_dist=uncovered_dist / uncovered_dist.sum(),
                            y=np.array([self.classes_[default_class_idx]]),
                            y_class_index=default_class_idx,
                            classes=self.classes_, n_samples=uncovered_dist)
        if not all_covered:
            default_rule.cov = uncovered_instances.sum() / self.X_.shape[0]
            default_rule.conf = uncovered_dist[
                                    default_class_idx] / uncovered_instances.sum()
            default_rule.supp = uncovered_dist[default_class_idx] / \
                                self.X_.shape[0]
        ruleset.rules.append(default_rule)
        return True

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from
        each iteration cycle

        :param combined_rules: the combined rules that are obtained on each
        iteration of the combination cycle

        :return:the ruleset with best performance in accordance to the metric
        parameter
        """
        rule_added = self._add_default_rule(combined_rules)
        combined_rules.compute_classification_performance(self.X_, self.y_,
                                                          self.metric)
        if rule_added:
            combined_rules.rules.pop()

        if combined_rules.metric(self.metric) > simplified_ruleset.metric(
                self.metric):
            self._early_stop_cnt = 0
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Combined rules: {combined_rules.metric(self.metric)}')
            return combined_rules, 'comb'
        else:
            self._early_stop_cnt += 1
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Previous combined rules: {simplified_ruleset.metric(self.metric)}')
            return simplified_ruleset, 'simp'

    def _weight_function(self, n):
        if n == 0:
            return 0
        if self.m is None:
            return n / (n + n)
        else:
            return n / (n + self.m)
    # def _get_gbm_init(self):
    #     """ get the initial estimate of a GBM ensemble
    #
    #     :return:
    #     """
    #     if isinstance(self.base_ensemble, GradientBoostingClassifier):
    #         return self.base_ensemble._raw_predict_init(self.X_[0].reshape(1, -1))
    #     if isinstance(self.base_ensemble, XGBClassifier):
    #         return self.base_ensemble.base_score
    #     return 0.0
