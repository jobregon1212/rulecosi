"""
This is a module to be used as a reference for building other modules
"""
import operator
from functools import reduce

from bitarray import bitarray
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
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
    if isinstance(ensemble, (AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)):
        return 'classifier'
    elif isinstance(ensemble, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        return 'gradient_classifier'
    else:
        return 'regressor'


def _ensemble_has_weights(ensemble):
    if isinstance(ensemble, (AdaBoostClassifier, AdaBoostRegressor)):
        return True
    else:
        return False


def _pessimistic_error_rate(N, e, z_alpha_half):
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(
        ((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N ** 2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator / denominator


class BaseRuleCOSI(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    # """

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 rule_max_depth=10,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 rule_order='cov',
                 early_stop=0.30,
                 metric='gmean',
                 column_names=None,
                 random_state=None):

        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.rule_max_depth = rule_max_depth
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.rule_order = rule_order
        self.early_stop = early_stop
        self.metric = metric
        self.random_state = random_state

        self._base_ens_type = None
        self.classes_ = None
        self._weights = None
        self.column_names = column_names

        self.X_ = None
        self.y_ = None

        self.original_X_ = None
        self.original_y_ = None

        self.original_rulesets = None
        self.simplified_ruleset = None
        # self._baseline_rulesets = None

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
            raise ValueError("Parameter n_estimators should be at least 2 to use RuleCOSI method.")

    def fit(self, X, y, sample_weight=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
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
        # If the ensemble has weights that are different for each tree, the trees are ordered according their weights
        if _ensemble_has_weights(self.base_ensemble):
            self._weights = copy.copy(self.base_ensemble.estimator_weights_)
            # check if weights are the same
            if np.max(self._weights) != np.min(self._weights):
                # order the rulesets according weight
                self._weights, processed_rulesets = helpers.order_trees_by_weight(processed_rulesets,
                                                                                  self._weights)
            else:
                self._weights = None

        # Then the duplicated rules are removed
        # processed_rulesets, self._weights, _ = helpers.remove_duplicated_rules(processed_rulesets,
        #                                                                        self._weights)

        # self._baseline_rulesets = processed_rulesets

        self._initialize_sets()
        self.simplified_ruleset = processed_rulesets[0]
        self._compute_rule_measures(self.simplified_ruleset)
        if isinstance(self.base_ensemble, CatBoostClassifier):
            self._add_default_rule(self.simplified_ruleset)
        self.simplified_ruleset.compute_classification_performance(self.X_, self.y_)

        self._n_combinations = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(self.simplified_ruleset)

        self.original_X_ = np.copy(self.X_)
        self.original_y_ = np.copy(self.y_)
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset, processed_rulesets[i])
            # order by rule confidence
            self._sort_ruleset(combined_rules)
            # prune inaccurate rules
            #print('Iteration {} - combinedrules: {} -- {} combinations'.format(i, helpers.count_rules_conds(combined_rules),self._n_combinations))
            self._compute_rule_measures(combined_rules, sequential_coverage=True)
            #print('Iteration {} - pruned rules: {} -- {} combinations'.format(i, helpers.count_rules_conds(combined_rules), self._n_combinations))
            # optimize rules
            self._simplify_rulesets(combined_rules)
            if len(combined_rules.rules) == 0:
                continue
            self.simplified_ruleset = self._evaluate_combinations(self.simplified_ruleset, combined_rules)
            #print('Iteration {} simplified rules: {} -- {} combinations'.format(i, helpers.count_rules_conds(combined_rules),self._n_combinations))
            if self._early_stop_cnt >= early_stop:
                break

        self._add_default_rule(self.simplified_ruleset)
        end_time = time.time()
        self._combination_time = end_time - start_time

    def _validate_and_create_base_ensemble(self):
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
        elif isinstance(self.base_ensemble, (AdaBoostClassifier, BaggingClassifier)):
            if is_classifier(self):
                self.base_ensemble.base_estimator = DecisionTreeClassifier(max_depth=self.tree_max_depth)
            else:
                self.base_ensemble.base_estimator = DecisionTreeRegressor(max_depth=self.tree_max_depth)
        else:
            self.base_ensemble.max_depth = self.tree_max_depth
        self._base_ens_type = _ensemble_type(self.base_ensemble)
        return clone(self.base_ensemble)

    def print_rules(self, verbose=0, return_string=False):
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

    # def _test_print_rules(self, rulelist):
    #     return_str = 'cov \tconf \tsupp \trule\n'
    #     i = 1
    #     for rule in rulelist:
    #         rule_string = '{:.4f}\t{:.4f}\t{:.4f}\tr_{}: '.format(rule.cov, rule.conf, rule.supp, i)
    #         rule_string = rule_string + ' ˄ '.join(map(lambda cond: str(self._global_condition_map[cond]), rule.A))
    #         rule_string += ' → ' + str(rule.y)
    #         rule_string += '\n'
    #         return_str += rule_string
    #         i += 1
    #     print(return_str)

    def _initialize_sets(self):
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._training_bit_sets = self._compute_training_bit_sets()
        self._cond_cov_dict = self._compute_condition_bit_sets()

    def _sort_ruleset(self, ruleset):
        if self.rule_order == 'cov':
            ruleset.rules.sort(key=lambda rule: (rule.cov, len(rule.A) * -1, rule.conf), reverse=True)
        elif self.rule_order == 'conf':
            ruleset.rules.sort(key=lambda rule: (rule.conf, len(rule.A) * -1, rule.cov), reverse=True)

    def _compute_condition_bit_sets(self):
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
        return [bitarray((self.y_ == self.classes_[i]).astype(int).tolist()) for i in range(len(self.classes_))]

    # main guiding procedure for combining rulesets for classification, make a combination of each of the class with
    # itself and all the other classes
    def _combine_rulesets(self, ruleset1, ruleset2):
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if (rule1.y == [class_one])]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if (rule2.y == [class_two])]
                combined_rules.update(self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        return RuleSet(list(combined_rules), self._global_condition_map)

    # actual combination procedure between to class-sliced rulesets
    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A) == 0 or len(r2.A) == 0:
                    continue
                self._n_combinations += 1  # count the actual number of combinations
                r1_AUr2_A = set(r1.A).union(r2.A)

                # create the new rule and compute class distribution and predicted class
                weight = None
                if self._base_ens_type == 'classifier':
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
                elif self._base_ens_type == 'gradient_classifier':
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
                    new_rule_cov, new_rule_conf_supp = self._get_conditions_measures(r1_AUr2_A)
                    new_rule.cov = new_rule_cov
                    new_rule.conf = new_rule_conf_supp[y_class_index][0]
                    new_rule.supp = new_rule_conf_supp[y_class_index][1]

                    if new_rule.cov > self.cov_threshold and \
                            new_rule.conf > self.conf_threshold: #and \
                            #len(new_rule.A) <= self.rule_max_depth:
                        combined_rules.add(new_rule)
                        self._good_combinations[new_rule] = [new_rule_cov,
                                                             new_rule_conf_supp[y_class_index][0],
                                                             new_rule_conf_supp[y_class_index][1]]
                    else:
                        self._bad_combinations.add(new_rule)

        return combined_rules

    # get the measures of individual conditions registered from the trees
    def _get_conditions_measures(self, conditions, uncovered_mask=None, return_set_size=False):
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
        #  return cov(r) ,[conf(r), supp(r)] - list by class
        coverage = cov_count / self.X_.shape[0]
        return coverage, [[class_count / cov_count, class_count / self.X_.shape[0]] for class_count in
                          class_cov_count]

    # simplify the conditions of a set of conditions, removing redundant conditions
    def _simplify_conditions(self, conditions, ):
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
            list_conds = [(int(id_), self._get_conditions_measures({int(id_)})[1][class_index]) for id_ in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(best_condition)  # remove the edge condition of the box from the list so it will remain
            [conditions.remove(cond[0]) for cond in list_conds]
        return frozenset(conditions)


    def _compute_rule_measures(self, combined_rules, sequential_coverage=False):
        uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
        if sequential_coverage:
            combined_rules.rules[:] = [rule for rule in combined_rules if
                                       self._rule_is_accurate(rule, uncovered_instances)]
        else:
            [rule for rule in combined_rules if self._rule_is_accurate(rule, uncovered_instances)]
        # print(uncovered_instances.count())
        # for rule in combined_rules:
        #     if uncovered_instances.count() < 50:
        #         stop=[]
        #     if not self._rule_is_accurate(rule, uncovered_instances):
        #         combined_rules.remove(rule)

    def _rule_is_accurate(self, rule, uncovered_instances):
        if uncovered_instances.count() == 0:
            return False
        local_uncovered_instances = copy.copy(uncovered_instances)
        rule_cov, rule_conf_supp = self._get_conditions_measures(rule.A, uncovered_mask=local_uncovered_instances)
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

    # def _test_ruleset(self, ruleset):
    #     return RuleSet(list(ruleset), self._global_condition_map)
    #
    # def _test_conds(self, conditions):
    #     return [self._global_condition_map[cond] for cond in conditions]

    # simplification method using pessimist error
    def _simplify_rulesets(self, combined_rules):
        for rule in combined_rules:
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

        combined_rules.rules[:] = [rule for rule in combined_rules if len(rule.A) > 0]
        # combined_rules.sort(key=lambda r: (r.cov, r.conf), reverse=True)
        self._sort_ruleset(combined_rules)
        self._compute_rule_measures(combined_rules, sequential_coverage=True)

    def _compute_pessimistic_error(self, conditions, class_index):
        if len(conditions) == 0:
            e = (self.X_.shape[0] - self._training_bit_sets[class_index].count()) / self.X_.shape[0]
            return 100 * _pessimistic_error_rate(self.X_.shape[0], e, 1.15)
        cov, class_cov = self._get_conditions_measures(conditions, return_set_size=True)
        total_instances = cov
        accurate_instances = class_cov[class_index]

        error_instances = total_instances - accurate_instances
        alpha_half = 1.15  # 25 % confidence for C4.5
        e = error_instances / total_instances  # totalInstances

        return 100 * _pessimistic_error_rate(total_instances, e, alpha_half)

    # def _add_default_rule(self, simplified_ruleset):
    #     uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
    #     [rule for rule in simplified_ruleset if self._rule_covers(rule, uncovered_instances)]
    #
    #     all_covered = False
    #     if uncovered_instances.count() == 0:
    #         uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
    #         all_covered = True
    #
    #     uncovered_labels = self.y_[uncovered_instances.tolist()]
    #     uncovered_dist = np.array([(uncovered_labels == class_).sum() for class_ in self.classes_])
    #     default_class_idx = np.argmax(uncovered_dist)
    #     default_rule = Rule({}, class_dist=uncovered_dist / uncovered_dist.sum(),
    #                         y=np.array([self.classes_[default_class_idx]]), y_class_index=default_class_idx,
    #                         classes=self.classes_)
    #     if not all_covered:
    #         default_rule.cov = uncovered_instances.count() / self.X_.shape[0]
    #         default_rule.conf = uncovered_dist[default_class_idx] / uncovered_instances.count()
    #         default_rule.supp = uncovered_dist[default_class_idx] / self.X_.shape[0]
    #     #     default_rule.set_measures(cov=uncovered_instances.count() / self.X_.shape[0],
    #     #                               conf=uncovered_dist[default_class_idx] / uncovered_instances.count(),
    #     #                               supp=uncovered_dist[default_class_idx] / self.X_.shape[0])
    #     # simplified_ruleset.append(default_rule)
    #     return True

    def _add_default_rule(self, simplified_ruleset):

        uncovered_instances = ~simplified_ruleset._predict(self.X_)[1]

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
        simplified_ruleset.rules.append(default_rule)
        return True

    def _rule_covers(self, rule, uncovered_instances):
        local_uncovered_instances = copy.copy(uncovered_instances)
        _, _ = self._get_conditions_measures(rule.A, uncovered_mask=local_uncovered_instances)
        uncovered_instances.clear()
        uncovered_instances.extend(local_uncovered_instances)

    # def _evaluate_combinations(self, simplified_ruleset, combined_rules):
    #     rule_added = self._add_default_rule(simplified_ruleset)
    #     sim_rules_perf = f1_score(self.y_,
    #                               RuleSet(simplified_ruleset, self._global_condition_map).predict(self.X_),
    #                               average='weighted')
    #     if rule_added:
    #         simplified_ruleset.pop()
    #
    #     rule_added = self._add_default_rule(combined_rules)
    #     comb_rules_perf = f1_score(self.y_,
    #                                RuleSet(combined_rules, self._global_condition_map).predict(self.X_),
    #                                average='weighted')
    #     if rule_added:
    #         combined_rules.pop()
    #
    #     if comb_rules_perf > sim_rules_perf:
    #         #print('winner gmean comb', comb_rules_perf)
    #         return combined_rules
    #     else:
    #         #print('winner gmean sim', sim_rules_perf)
    #         return simplified_ruleset

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):

        # simplified_ruleset.pop()
        rule_added = self._add_default_rule(combined_rules)
        combined_rules.compute_classification_performance(self.X_, self.y_, self.metric)
        if rule_added:
            combined_rules.rules.pop()

        if combined_rules.metric(self.metric) >= simplified_ruleset.metric(self.metric):
            #print('winner {} combined rules {}'.format(self.metric, combined_rules.metric(self.metric)))
            self._early_stop_cnt = 0
            return combined_rules
        else:
            self._early_stop_cnt += 1
            #print('early_stop_counter = {}. Winner {} simplified rules {}'.format(self._early_stop_cnt, self.metric, simplified_ruleset.metric(self.metric)))
            return simplified_ruleset

    def _get_gbm_init(self):
        if isinstance(self.base_ensemble, GradientBoostingClassifier):
            return self.base_ensemble._raw_predict_init(self.X_[0].reshape(1, -1))
        if isinstance(self.base_ensemble, XGBClassifier):
            return self.base_ensemble.base_score
        return 0.0


class RuleCOSIClassifier(ClassifierMixin, BaseRuleCOSI):
    """ An example classifier which implements a 1-NN algorithm.

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

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

        return self

    # _clf = AdaBoostClassifier()

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.simplified_ruleset.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        return self.simplified_ruleset.predict_proba(X)

    # def predict_baseline(self, X, proba=False):
    #     check_is_fitted(self, ['X_', 'y_'])
    #
    #     # Input validation
    #     X = check_array(X)
    #     if proba:
    #         return np.array([self.predict_baseline_single(row, proba=True) for row in X])
    #     else:
    #         return np.array([self.predict_baseline_single(row) for row in X])
    #
    # def predict_baseline_single(self, X, proba=False):
    #     avg_class_dist = self._baseline_rulesets[0].get_rule_list()[0].class_dist * 0
    #     sum_leaf_score = None
    #     if self._base_ens_type == 'gradient_classifier':
    #         sum_leaf_score = self._get_gbm_init()
    #     for idx, ruleset in enumerate(self._baseline_rulesets):
    #         for rule in ruleset:
    #             if rule.covers(X, ruleset.get_condition_map()):
    #                 if self._base_ens_type == 'classifier':
    #                     if self._weights is None:
    #                         avg_class_dist = np.mean([avg_class_dist, rule.class_dist],
    #                                                  axis=0).reshape((len(self.classes_),))
    #                 elif self._base_ens_type == 'gradient_classifier':
    #                     sum_leaf_score = rule.logit_score + sum_leaf_score
    #                 if proba:
    #                     return rule.class_dist
    #                 else:
    #                     return rule.y[0]
    #     if self._base_ens_type == 'classifier':
    #         y_class_index = np.argmax(avg_class_dist).item()
    #         y = [self.classes_[y_class_index]]
    #     elif self._base_ens_type == 'gradient_classifier':
    #         if len(self.classes_) == 2:
    #             raw_to_proba = expit(sum_leaf_score)
    #             avg_class_dist = np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])
    #         else:
    #             avg_class_dist = sum_leaf_score - logsumexp(sum_leaf_score)
    #         y_class_index = np.argmax(avg_class_dist).item()
    #         y = [self.classes_[y_class_index]]
    #     if proba:
    #         return avg_class_dist
    #     else:
    #         return y


class RuleCOSIRegressor(RegressorMixin, BaseRuleCOSI):
    """ An example classifier which implements a 1-NN algorithm.

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

    # _clf = AdaBoostClassifier()

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        super().fit(X, y)
        return self.base_ensemble

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.base_ensemble.predict(X)
