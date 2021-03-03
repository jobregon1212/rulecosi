"""RuleCOSI rule extractor.

This module contains the RuleCOSI extractor for classification problems.

The module structure is the following:

- The `BaseRuleCOSI` base class implements a common ``fit`` method
  for all the estimators in the module. This is done because hopefully in the future,
  the algorithm will work with regression problems as well.

- :class:`rulecosi.RuleCOSIClassifier` implements rule extraction from a variety of ensembles
   for classification problems.

"""

# Authors: Josue Obregon <jobregon@khu.ac.kr>
#
#
# License: TBD
import operator
from abc import abstractmethod, ABCMeta
from functools import reduce

from bitarray import bitarray
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier, is_regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import copy
import time

from ast import literal_eval
from math import sqrt
from rulecosi.rules import Rule, RuleSet
from scipy.special import expit, logsumexp
import rulecosi.helpers as helpers
from rulecosi.rule_extraction import get_rule_extractor


def _ensemble_type(ensemble):
    """ Return the ensemble type


    :param ensemble:
    :return:
    """
    if isinstance(ensemble, (BaggingClassifier, RandomForestClassifier)):
        return 'bagging'
    elif isinstance(ensemble, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        return 'gbt'
    else:
        return 'gbt'


def _pessimistic_error_rate(N, e, z_alpha_half):
    """ Computes a statistical correction to the training error to estimate the generalization error.

    This function assumes that the errors on a rule (leaf node) follow a binomial distribution. Therefore, it
    computes the statistical correction as the upper limit of the normal approximation of a binomial distribution
    of the training error e.

    :param N: number of training records
    :param e: training error
    :param z_alpha_half: standardized value from a standard normal distribution
    :return:
    """
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(
        ((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N ** 2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator / denominator


class BaseRuleCOSI(BaseEstimator, metaclass=ABCMeta):
    """ Base class for RuleCOSI estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 min_samples=1,
                 early_stop=0.30,
                 metric='gmean',
                 column_names=None,
                 random_state=None):

        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.cov_threshold = cov_threshold
        self.min_samples = min_samples
        self.metric = metric
        self.early_stop = early_stop
        self.random_state = random_state

        self._base_ens_type = None
        self.classes_ = None
        self._weights = None
        self.column_names = column_names

        self.X_ = None
        self.y_ = None

        self.original_rulesets = None
        self.simplified_ruleset = None

        self._global_condition_map = None
        self._cond_cov_dict = None
        self._training_bit_sets = None

        self._bad_combinations = None
        self._good_combinations = None

        self._n_combinations = 0
        self._ensemble_training_time = 0
        self._combination_time = 0
        self._early_stop_cnt = 0

        if self.n_estimators < 2:
            raise ValueError("Parameter n_estimators should be at least 2 to use the RuleCOSI method.")

    def fit(self, X, y, sample_weight=None):
        """ Combine and simplify the decision trees from the base ensemble and builds a rule-based classifier using the
            training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        sample_weight: Currently this is not supported and it is here just for compatibility reasons

        Returns
        -------
        self : object
        """

        # Check that X and y have correct shape
        if self.column_names is None:
            if isinstance(X, pd.DataFrame):
                self.column_names = X.columns

        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.base_ensemble = self._validate_and_create_base_ensemble()
        start_time = time.time()
        self.base_ensemble.fit(X, y, sample_weight)
        end_time = time.time()
        self._ensemble_training_time = end_time - start_time
        start_time = time.time()

        # First step is extract the rules
        extractor = get_rule_extractor(self.base_ensemble, self.column_names, self.classes_, self.X_)
        self.original_rulesets, self._global_condition_map = extractor.extract_rules()
        processed_rulesets = copy.deepcopy(self.original_rulesets)

        self._initialize_sets()
        self.simplified_ruleset = processed_rulesets[0]
        self._compute_rule_heuristics(self.simplified_ruleset)
        if isinstance(self.base_ensemble, CatBoostClassifier):
            self._add_default_rule(self.simplified_ruleset)
        self.simplified_ruleset.compute_classification_performance(self.X_, self.y_)

        self._n_combinations = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)
        no_combinations = True  # flag to control if there are no combinations registered
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset, processed_rulesets[i])
            # prune inaccurate rules
            self._sequential_covering_pruning(combined_rules)
            # simplify rules
            self._simplify_rulesets(combined_rules)
            # skip if the combined rules are empty
            if len(combined_rules.rules) == 0:
                continue
            self.simplified_ruleset = self._evaluate_combinations(self.simplified_ruleset, combined_rules)

            if self._early_stop_cnt >= early_stop:
                break
            if self.simplified_ruleset.metric() == 1:
                break
            no_combinations = False

        if no_combinations:  # if any combination was successful, we just simplify the first ruleset
            self._simplify_rulesets(self.simplified_ruleset)
        self._add_default_rule(self.simplified_ruleset)
        end_time = time.time()
        self._combination_time = end_time - start_time

    def _validate_and_create_base_ensemble(self):
        """ Validate the parameter of base ensemble and if it is None, it set the default ensemble,
            GradientBoostingClassifier.

        """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))
        if self.base_ensemble is None:
            if is_classifier(self):
                self.base_ensemble = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                                max_depth=self.tree_max_depth,
                                                                random_state=self.random_state)
            elif is_regressor(self):
                self.base_ensemble = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                               max_depth=self.tree_max_depth,
                                                               random_state=self.random_state)
            else:
                raise ValueError("You should choose an original classifier/regressor ensemble to use RuleCOSI method.")
        self.base_ensemble.n_estimators = self.n_estimators
        if isinstance(self.base_ensemble, CatBoostClassifier):
            self.base_ensemble.set_params(n_estimators=self.n_estimators, depth=self.tree_max_depth)
        elif isinstance(self.base_ensemble, BaggingClassifier):
            if is_classifier(self):
                self.base_ensemble.base_estimator = DecisionTreeClassifier(max_depth=self.tree_max_depth)
            else:
                self.base_ensemble.base_estimator = DecisionTreeRegressor(max_depth=self.tree_max_depth)
        else:
            self.base_ensemble.max_depth = self.tree_max_depth
        self._base_ens_type = _ensemble_type(self.base_ensemble)
        return clone(self.base_ensemble)

    @abstractmethod
    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the combination and simplification process
            This includes the set of good combinations G and bad combinations B, but can include other sets
            necessary to the combination process
        """
        pass

    @abstractmethod
    def _compute_rule_heuristics(self, ruleset):
        """ Compute the rule heuristics for each of the rules in the ruleset

        :param ruleset: ruleset R to be measured
        """
        pass

    @abstractmethod
    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different criteria

        :param ruleset: ruleset R to which the default rule will be added
        """
        pass

    @abstractmethod
    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using the procedure described in the paper [ref]

        :param ruleset1: ruleset 1 to be combined
        :param ruleset2: ruleset 2 to be combined
        :return: a new ruleset containing the result of the combination process
        """
        pass

    @abstractmethod
    def _sequential_covering_pruning(self, ruleset):
        """ Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it sorts it and find the best rule. Then the
        covered instances are removed from the training set and the process is repeated until one of three stopping
        criteria are met: 1. all the records of the training set are covered, 2. all the rules on ruleset are used or
        3. there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        pass

    @abstractmethod
    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset using the pessimist error.

        The function simplify the ruleset by iteratively removing conditions that minimize the pessimistic error.
        If all the conditions of a rule are removed, then the rule is discarded.

        :param ruleset:
        :return:
        """
        pass

    @abstractmethod
    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from each iteration cycle
        :param combined_rules: the combined rules that are obtained on each iteration of the combination cycle
        :return:the ruleset with best performance in accordance to the metric parameter
        """
        pass


class RuleCOSIClassifier(ClassifierMixin, BaseRuleCOSI):
    """ A simplified rule-based classifier based on an ensemble of trees

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstration of how to pass and store parameters.


    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 min_samples=1,
                 early_stop=0.30,
                 metric='gmean',
                 column_names=None,
                 random_state=None,
                 conf_threshold=0.5,
                 rule_order='cov'
                 ):

        super().__init__(base_ensemble=base_ensemble,
                         n_estimators=n_estimators,
                         tree_max_depth=tree_max_depth,
                         cov_threshold=cov_threshold,
                         min_samples=min_samples,
                         early_stop=early_stop,
                         metric=metric,
                         column_names=column_names,
                         random_state=random_state,
                         )


        self.conf_threshold = conf_threshold
        self.rule_order = rule_order

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """ Predict classes for X.

        The predicted class of an input sample. The prediction use the simplified ruleset and evaluate the rules
        one by one. When a rule covers a sample, the head of the rule is returned as predicted class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class. The class with the highest value in the class distribution of the fired rule.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.simplified_ruleset.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is obtained from the class distribution of the fired rule.
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
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        return self.simplified_ruleset.predict_proba(X)

    def print_rules(self, verbose=0, return_string=False):
        """ Print the rules in a string format. It can also print rule heuristics depending on the verbose parameter.

        :param verbose: 0 for only printing the rules, 1 for printing also the rule heuristics
        :param return_string: if True, the rules are returned as string and no printing is done. Default is False.
        :return:
        """
        check_is_fitted(self, ['X_', 'y_'])

        return_str = str(self.simplified_ruleset)

        if verbose == 1:
            return_str = 'cov \tconf \tsupp \trule\n'
            i = 1
            for rule in self.simplified_ruleset:
                rule_string = '{:.4f}\t{:.4f}\t{:.4f}\tr_{}: '.format(rule.cov, rule.conf, rule.supp, i)
                if len(rule.A) == 0:
                    rule_string = rule_string + '( )'
                else:
                    rule_string = rule_string + ' ˄ '.join(
                        map(lambda cond: str(self._global_condition_map[cond]), rule.A))
                rule_string += ' → ' + str(rule.y)
                rule_string += '\n'
                return_str += rule_string
                i += 1
        if return_string:
            return return_str
        else:
            print(return_str)

    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the combination and simplification process
            This includes the set of good combinations G and bad combinations B. It also includes the bitsets for
            the training data as well as the bitsets for each of the conditions
        """
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._training_bit_sets = self._compute_training_bit_sets()
        self._cond_cov_dict = self._compute_condition_bit_sets()

    def _sort_ruleset(self, ruleset):
        """ Sort the ruleset in place according to the rule_order parameter.

        :param ruleset: ruleset to be ordered
        """
        if self.rule_order == 'cov':
            ruleset.rules.sort(key=lambda rule: (rule.cov, rule.conf, rule.supp), reverse=True)

        elif self.rule_order == 'conf':
            ruleset.rules.sort(key=lambda rule: (rule.conf, rule.cov, rule.supp), reverse=True)

    def _compute_condition_bit_sets(self):
        """ Compute the bitsets of the coverage of every condition contained in the ensemble according to the
        training set

        """
        # empty sets for each condition coverage class
        cond_cov_dict = [{} for _ in range(len(self.classes_) + 1)]
        for cond_id, cond in self._global_condition_map.items():
            # compute bitarray for the covered records in X_ by condition cond
            cond_coverage_bitarray = bitarray(cond.satisfies_array(self.X_).astype(int).tolist())
            # create the entries in the dictionary
            for i in range(len(self.classes_)):
                cond_cov_dict[i][cond_id] = cond_coverage_bitarray & self._training_bit_sets[i]
            cond_cov_dict[-1][cond_id] = cond_coverage_bitarray
        return cond_cov_dict

    def _compute_training_bit_sets(self):
        """ Compute the bitsets of the coverage for the prior class distribution of the ensemble according to the
        training set

        """
        return [bitarray((self.y_ == self.classes_[i]).astype(int).tolist()) for i in range(len(self.classes_))]

    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using the procedure described in the paper [ref]

        Main guiding procedure for combining rulesets for classification, make a combination of each of the class with
        itself and all the other classes

        :param ruleset1:
        :param ruleset2:
        :return: ruleset containing the combination of ruleset1 and ruleset 2
        """
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if (rule1.y == [class_one])]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if (rule2.y == [class_two])]
                combined_rules.update(self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        return RuleSet(list(combined_rules), self._global_condition_map)

    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        """ Actual combination procedure between to class-sliced rulesets

        :param s_ruleset1: sliced ruleset 1 according to a class
        :param s_ruleset2: sliced ruleset according to a class
        :return: a set of rules containing the combined rules of s_ruleset1 and s_ruleset2
        """
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A) == 0 or len(r2.A) == 0:
                    continue
                self._n_combinations += 1  # count the actual number of combinations
                r1_AUr2_A = set(r1.A).union(r2.A)

                # create the new rule and compute class distribution and predicted class
                weight = None
                if self._base_ens_type == 'bagging':
                    if self._weights is None:
                        class_dist = np.mean([r1.class_dist, r2.class_dist],
                                             axis=0).reshape((len(self.classes_),))
                    else:
                        class_dist = np.average([r1.class_dist, r2.class_dist], axis=0,
                                                weights=[r1.weight, r2.weight]).reshape((len(self.classes_),))
                        weight = (r1.weight() + r2.weight) / 2
                    y_class_index = np.argmax(class_dist).item()
                    y = np.array([self.classes_[y_class_index]])
                    logit_score = 0
                elif self._base_ens_type == 'gbt':
                    logit_score = r1.logit_score + r2.logit_score
                    if len(self.classes_) == 2:
                        raw_to_proba = expit(logit_score)
                        class_dist = np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])
                    else:
                        class_dist = logit_score - logsumexp(logit_score)
                    y_class_index = np.argmax(class_dist).item()
                    y = np.array([self.classes_[y_class_index]])
                elif self._base_ens_type == 'regressor':
                    y = np.mean([r1.y, r2.y], axis=0)

                if isinstance(self.base_ensemble, CatBoostClassifier):
                    self._remove_opposite_conditions(r1_AUr2_A, y_class_index)

                n_samples = np.sum([r1.n_samples, r2.n_samples], axis=0)
                new_rule = Rule(frozenset(r1_AUr2_A), class_dist=class_dist, logit_score=logit_score, y=y,
                                y_class_index=y_class_index, n_samples=n_samples, classes=self.classes_, weight=weight)

                # check if the combination was null before, if it was we just skip it
                if new_rule in self._bad_combinations:
                    continue
                # if the combination was a good one before, we just add the combination to the rules
                if new_rule in self._good_combinations:
                    good_rule_measures = self._good_combinations[new_rule]
                    new_rule.cov = good_rule_measures[0]
                    new_rule.conf = good_rule_measures[1]
                    new_rule.supp = good_rule_measures[2]
                    # new_rule.set_measures(good_rule_measures[0], good_rule_measures[1], good_rule_measures[1])
                    combined_rules.add(new_rule)
                else:
                    new_rule_cov, new_rule_conf_supp = self._get_conditions_heuristics(r1_AUr2_A)
                    new_rule.cov = new_rule_cov
                    new_rule.conf = new_rule_conf_supp[y_class_index][0]
                    new_rule.supp = new_rule_conf_supp[y_class_index][1]

                    if new_rule.cov > self.cov_threshold and \
                            new_rule.conf > self.conf_threshold:
                        combined_rules.add(new_rule)
                        self._good_combinations[new_rule] = [new_rule_cov,
                                                             new_rule_conf_supp[y_class_index][0],
                                                             new_rule_conf_supp[y_class_index][1]]
                    else:
                        self._bad_combinations.add(new_rule)

        return combined_rules

    def _get_conditions_heuristics(self, conditions, uncovered_mask=None, return_set_size=False):
        """ Compute the heuristics of the combination of conditions using the bitsets  of each condition
        from the training set. An intersection operation is made and the cardinality of the resultant set is used
        for computing the heuristics

        :param conditions: set of conditions' id
        :param uncovered_mask: if different than None, mask out the records that are already covered from
                               the training set. Default is None.
        :param return_set_size: If True, return the size of the set of covered records instead of the sets. Default is
                                False
        :return: the bitsets representing the coverage, coverage of positive and coverage of negative classes
        """
        if len(conditions) == 0:
            return 0.0, np.zeros((self.classes_.shape[0], 2))
        b_array_conds = [reduce(operator.and_, [self._cond_cov_dict[i][cond] for cond in conditions])
                         for i in range(len(self.classes_))]
        b_array_conds.append(reduce(operator.or_, [i for i in b_array_conds]))

        if uncovered_mask is not None:
            # cov_conf_b_arrays = [b_array_measure & mask for b_array_measure in b_array_conds]
            b_array_conds = [b_array_measure & uncovered_mask for b_array_measure in b_array_conds]

            updated_mask = ~b_array_conds[-1] & uncovered_mask
            uncovered_mask.clear()
            uncovered_mask.extend(updated_mask)
        cov_count = b_array_conds[-1].count()
        if cov_count == 0:
            return 0.0, np.zeros((self.classes_.shape[0], 2))

        class_cov_count = [b_array_conds[i].count() for i in range(len(self.classes_))]
        if return_set_size:
            return b_array_conds[-1].count(), class_cov_count
        coverage = cov_count / self.X_.shape[0]
        return coverage, [[class_count / cov_count, class_count / self.X_.shape[0]] for class_count in
                          class_cov_count]

    def _simplify_conditions(self, conditions):
        """ Remove redundant conditions of a single rule.

        Redundant conditions are conditions with the same attribute and same operator but different value.

        For example: (att1 > 5) ^ (att1 > 10). In this case we keep the second one because it contains the first one

        :param conditions: set of conditions' ids
        :return: return the set of conditions with no redundancy
        """
        cond_map = self._global_condition_map  # just for readability
        # create list with this format ["(att_index, 'OPERATOR')", 'cond_id']
        att_op_list = [[str((cond_map[cond].att_index, cond_map[cond].op.__name__)), cond]
                       for cond in conditions]
        att_op_list = np.array(att_op_list)
        # First part is to remove redundant conditions (e.g. att1>5 and att1> 10)
        # create list for removing redundant conditions
        dict_red_cond = {str(i[0]): att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 1]
                         for i in att_op_list}
        # create generator to traverse just conditions with the same att_index and operator that appear more than once
        gen_red_cond = ((att_op, conds) for (att_op, conds) in dict_red_cond.items() if len(conds) > 1)

        for (att_op, conds) in gen_red_cond:
            tup_att_op = literal_eval(att_op)
            list_conds = {cond_map[int(id_)] for id_ in conds}
            if tup_att_op[1] in ['lt', 'le']:
                edge_condition = max(list_conds, key=lambda item: item.value)  # condition at the edge of the box
            if tup_att_op[1] in ['gt', 'ge']:
                edge_condition = min(list_conds, key=lambda item: item.value)
            list_conds.remove(edge_condition)  # remove the edge condition of the box from the list, so it will remain
            [conditions.remove(hash(cond)) for cond in list_conds]

        return frozenset(conditions)

    def _remove_opposite_conditions(self, conditions, class_index):
        """ Removes conditions that have disjoint regions and will make a rule to be discarded because it would have
            null coverage.

            This function is used with the trees generated with CatBoost algorithm, which are called oblivious trees.
            This trees share the same splits among entire levels. So The rules generated when combining tend to create
            many rules with null coverage, so this function helps to avoid this problem and explore better the covered
            feature space combination.

        :param conditions: set of conditions' ids
        :param class_index: predicted class of the rule created with the set of conditions
        :return: set of conditions with no opposite conditions
        """
        att_op_list = [[self._global_condition_map[cond].att_index, self._global_condition_map[cond].op.__name__, cond]
                       for cond in conditions]
        att_op_list = np.array(att_op_list)
        # Second part is to remove opposite operator conditions (e.g. att1>=5  att1<5)
        # create list for removing opposite conditions
        dict_opp_cond = {str(i[0]): att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 2]
                         for i in att_op_list}
        # create generator to traverse just conditions with the same att_index and different operator that appear
        # more than once
        gen_opp_cond = ((att, conds) for (att, conds) in dict_opp_cond.items() if len(conds) > 1)
        for (_, conds) in gen_opp_cond:
            # att_index = literal_eval(att)
            list_conds = [(int(id_), self._get_conditions_heuristics({int(id_)})[1][class_index]) for id_ in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(best_condition)  # remove the edge condition of the box from the list so it will remain
            [conditions.remove(cond[0]) for cond in list_conds]
        return frozenset(conditions)

    def _compute_rule_heuristics(self, ruleset, sequential_coverage=False):
        """ Compute the rule heuristics for each of the rules in the ruleset and update its values accordingly.

        :param ruleset: RuleSet object representing a ruleset
        :param sequential_coverage: If true, the covered examples covered by one rule are removed. Additionally, if a
                                    rule does not meet the threshold is discarded. If false, it just compute the
                                    heuristics with all the records on the training set for all the rules. Default
                                    is False
        """
        uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
        if sequential_coverage:
            ruleset.rules[:] = [rule for rule in ruleset if
                                self._rule_is_accurate(rule, uncovered_instances)]
        else:
            [rule for rule in ruleset if self._rule_is_accurate(rule, uncovered_instances)]

    def _compute_rule_heuristics2(self, ruleset, uncovered_mask=None):
        """ Compute rule heuristics, but without the sequential_coverage parameter, and without removing the rules that
            do not meet the thresholds

        :param ruleset: RuleSet object representing a ruleset
        :param uncovered_mask: if different than None, mask out the records that are already covered from
                               the training set. Default is None.
        :return:
        """
        for rule in ruleset:
            local_uncovered_instances = copy.copy(uncovered_mask)
            rule_cov, rule_conf_supp = self._get_conditions_heuristics(rule.A, uncovered_mask=local_uncovered_instances)
            rule_conf = rule_conf_supp[rule.class_index][0]
            rule_supp = rule_conf_supp[rule.class_index][1]
            rule.cov = rule_cov
            rule.conf = rule_conf
            rule.supp = rule_supp

    def _sequential_covering_pruning(self, ruleset):
        """Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it sorts it and find the best rule. Then the
        covered instances are removed from the training set and the process is repeated until one of three stopping
        criteria are met: 1. all the records of the training set are covered, 2. all the rules on ruleset are used or
        3. there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        return_ruleset = []
        uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
        found_rule = True
        while len(ruleset.rules) > 0 and uncovered_instances.count() > 0 and found_rule:
            self._compute_rule_heuristics2(ruleset, uncovered_instances)
            self._sort_ruleset(ruleset)
            found_rule = False
            for rule in ruleset:
                if self._rule_is_accurate(rule, uncovered_instances=uncovered_instances):
                    return_ruleset.append(rule)
                    ruleset.rules[:] = [rule for rule in ruleset if rule != return_ruleset[-1]]
                    found_rule = True
                    break
        ruleset.rules[:] = return_ruleset

    def _rule_is_accurate(self, rule, uncovered_instances):
        """ Determine if a rule meet the coverage and confidence thresholds

        :param rule: a Rule object
        :param uncovered_instances:  mask out the records that are already covered from
                               the training set.
        :return:
        """
        if uncovered_instances.count() == 0:
            return False
        local_uncovered_instances = copy.copy(uncovered_instances)
        rule_cov, rule_conf_supp = self._get_conditions_heuristics(rule.A, uncovered_mask=local_uncovered_instances)
        rule_conf = rule_conf_supp[rule.class_index][0]
        rule_supp = rule_conf_supp[rule.class_index][1]
        rule.cov = rule_cov
        rule.conf = rule_conf
        rule.supp = rule_supp
        if rule_cov > self.cov_threshold and rule_conf > self.conf_threshold:
            uncovered_instances.clear()
            uncovered_instances.extend(local_uncovered_instances)
            return True
        else:
            return False

    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset inplace using the pessimist error.

        The function simplify the ruleset by iteratively removing conditions that minimize the pessimistic error.
        If all the conditions of a rule are removed, then the rule is discarded.

        :param ruleset:
        """
        for rule in ruleset:
            rule.A = self._simplify_conditions(set(rule.A))
            base_line_error = self._compute_pessimistic_error(rule.A, rule.class_index)
            min_error = 0
            while min_error <= base_line_error and len(rule.A) > 0:
                errors = [(cond, self._compute_pessimistic_error(rule.A.difference([cond]), rule.class_index))
                          for cond in rule.A]
                min_error_tup = min(errors, key=lambda tup: tup[1])
                min_error = min_error_tup[1]
                if min_error <= base_line_error:
                    base_line_error = min_error
                    min_error = 0
                    rule_conds = set(rule.A)
                    rule_conds.remove(min_error_tup[0])
                    rule.A = frozenset(rule_conds)

        min_cov = self.min_samples / self.X_.shape[0]
        ruleset.rules[:] = [rule for rule in ruleset if len(rule.A) > 0 and rule.cov > min_cov]
        # combined_rules.sort(key=lambda r: (r.cov, r.conf), reverse=True)

        # self._compute_rule_measures(combined_rules, sequential_coverage=False)
        self._compute_rule_heuristics(ruleset, sequential_coverage=False)
        self._sort_ruleset(ruleset)
        self._compute_rule_heuristics(ruleset, sequential_coverage=True)

    def _compute_pessimistic_error(self, conditions, class_index):
        """ Computes a statistical correction to the training error to estimate the generalization error of one rule.

        This function assumes that the errors on the rule follow a binomial distribution. Therefore, it
        computes the statistical correction as the upper limit of the normal approximation of a binomial distribution
        of the training error e.

        :param conditions: set of conditions' ids
        :param class_index: predicted class index of the rule
        :return: the statistical correction of the training error of that rule (between 0 and 100)
        """
        if len(conditions) == 0:
            e = (self.X_.shape[0] - self._training_bit_sets[class_index].count()) / self.X_.shape[0]
            return 100 * _pessimistic_error_rate(self.X_.shape[0], e, 1.15)
        cov, class_cov = self._get_conditions_heuristics(conditions, return_set_size=True)
        total_instances = cov
        accurate_instances = class_cov[class_index]

        error_instances = total_instances - accurate_instances
        alpha_half = 1.15  # 25 % confidence for C4.5
        e = error_instances / total_instances  # totalInstances

        return 100 * _pessimistic_error_rate(total_instances, e, alpha_half)

    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different criteria

        :param ruleset: ruleset R to which the default rule will be added
        """

        uncovered_instances = ~ruleset._predict(self.X_)[1]

        all_covered = False
        if uncovered_instances.sum() == 0:
            uncovered_dist = np.array([self._training_bit_sets[i].count() for i in range(len(self.classes_))])
            all_covered = True
        else:
            uncovered_labels = self.y_[uncovered_instances]
            uncovered_dist = np.array([(uncovered_labels == class_).sum() for class_ in self.classes_])

        default_class_idx = np.argmax(uncovered_dist)
        default_rule = Rule({}, class_dist=uncovered_dist / uncovered_dist.sum(),
                            y=np.array([self.classes_[default_class_idx]]), y_class_index=default_class_idx,
                            classes=self.classes_)
        if not all_covered:
            default_rule.cov = uncovered_instances.sum() / self.X_.shape[0]
            default_rule.conf = uncovered_dist[default_class_idx] / uncovered_instances.sum()
            default_rule.supp = uncovered_dist[default_class_idx] / self.X_.shape[0]
        ruleset.rules.append(default_rule)
        return True

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from each iteration cycle
        :param combined_rules: the combined rules that are obtained on each iteration of the combination cycle
        :return:the ruleset with best performance in accordance to the metric parameter
        """
        rule_added = self._add_default_rule(combined_rules)
        combined_rules.compute_classification_performance(self.X_, self.y_, self.metric)
        if rule_added:
            combined_rules.rules.pop()

        if combined_rules.metric(self.metric) >= simplified_ruleset.metric(self.metric):
            self._early_stop_cnt = 0
            return combined_rules
        else:
            self._early_stop_cnt += 1
            return simplified_ruleset

    def _get_gbm_init(self):
        """ get the initial estimate of a GBM ensemble

        :return:
        """
        if isinstance(self.base_ensemble, GradientBoostingClassifier):
            return self.base_ensemble._raw_predict_init(self.X_[0].reshape(1, -1))
        if isinstance(self.base_ensemble, XGBClassifier):
            return self.base_ensemble.base_score
        return 0.0
