"""
This is a module to be used as a reference for building other modules
"""
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import copy
from bitarray import bitarray, frozenbitarray
from bitarray.util import count_and
from ast import literal_eval
from math import sqrt
from rulecosi.rules import Condition, Rule, Operator, RuleSet
from scipy.special import expit, logsumexp
import rulecosi.helpers as helpers
from rulecosi._rule_extraction import get_rule_extractor


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
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N**2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator/denominator


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
                 random_state=None,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 max_rule_depth=5,
                 rule_order='cov'):
        self._base_ens = base_ensemble
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.max_rule_depth = max_rule_depth
        self.rule_order = rule_order

        self._base_ens_type = None
        self.classes_ = None
        self._weights = None
        self._column_names = None

        self.X_ = None
        self.y_ = None

        self._original_rulesets = None
        self.simplified_ruleset = None

        self._global_condition_map = None
        self._cond_cov_dict = None
        # self._training_bit_sets = None

        self._bad_combinations = None
        self._good_combinations = None

        self._n_combinations = 0


        if self.n_estimators < 2:
            raise ValueError("Parameter n_estimators should be at least 2 to use RuleCOSI method.")

    def fit(self, X, y, sample_weight=None, column_names=None):
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
        if column_names is None:
            if isinstance(X, pd.DataFrame):
                self._column_names = X.columns
        else:
            self._column_names = column_names

        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        if self._base_ens is None:
            if is_classifier(self):
                self._base_ens = AdaBoostClassifier(algorithm="SAMME",
                                                    n_estimators=self.n_estimators,
                                                    random_state=self.random_state)
            elif is_regressor(self):
                self._base_ens = AdaBoostRegressor(n_estimators=self.n_estimators,
                                                   random_state=self.random_state)
            else:
                if not is_classifier(self._base_ens):
                    raise ValueError("You should choose an original ensemble to use RuleCOSI method.")
        self._base_ens = clone(self._base_ens)

        self._base_ens.fit(X, y, sample_weight)

        # just for testing

        # self._base_ens.predict(X[0].reshape(1, -1))

        self._base_ens_type = _ensemble_type(self._base_ens)

        # First step is extract the rules
        extractor = get_rule_extractor(self._base_ens, self._column_names, self.classes_, self.X_)
        self._original_rulesets, self._global_condition_map = extractor.extract_rules()
        processed_rulesets = copy.deepcopy(self._original_rulesets)
        # If the ensemble has weights that are different for each tree, the trees are ordered according their weights
        if _ensemble_has_weights(self._base_ens):
            self._weights = copy.copy(self._base_ens.estimator_weights_)
            # check if weights are the same
            if np.max(self._weights) != np.min(self._weights):
                # order the rulesets according weight
                self._weights, processed_rulesets = helpers.order_trees_by_weight(processed_rulesets,
                                                                                  self._weights)
            else:
                self._weights = None

        # Then the duplicated rules are removed
        processed_rulesets, self._weights, _ = helpers.remove_duplicated_rules(processed_rulesets,
                                                                               self._weights)

        self._initialize_sets()
        self.simplified_ruleset = processed_rulesets[0].get_rule_list()
        self._compute_rule_measures(self.simplified_ruleset)

        self._n_combinations = 0

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

            self.simplified_ruleset = self._evaluate_combinations(self.simplified_ruleset, combined_rules)
            #print('Iteration {} simplified rules: {} -- {} combinations'.format(i, helpers.count_rules_conds(combined_rules),self._n_combinations))

            #test = RuleSet(list(combined_rules), self._global_condition_map)
            #null_test = RuleSet(list(self._bad_combinations), self._global_condition_map)
            #good_test = RuleSet(list(self._good_combinations), self._global_condition_map)
            #stop = ['stop']
        self._sort_ruleset(self.simplified_ruleset)
        self._add_default_rule(self.simplified_ruleset)
        self.simplified_ruleset = RuleSet(self.simplified_ruleset, self._global_condition_map)

    def print_rules(self, verbose=0):
        check_is_fitted(self, ['X_', 'y_'])
        if verbose==0:
            print(self.simplified_ruleset)
            return
        if verbose==1:
            return_str = 'cov \tconf \tsupp \trule\n'
            i = 1
            for rule in self.simplified_ruleset.get_rule_list():
                rule_string = '{:.4f}\t{:.4f}\t{:.4f}\tr_{}: '.format(rule.cov(),rule.conf(),rule.supp(),i)
                rule_string = rule_string + ' ˄ '.join(map(lambda cond: str(self._global_condition_map[cond]), rule._A))
                rule_string += ' → ' + str(rule._y)
                rule_string += '\n'
                return_str += rule_string
                i += 1
            print(return_str)

    def _test_print_rules(self, rulelist ):
        return_str = 'cov \tconf \tsupp \trule\n'
        i = 1
        for rule in rulelist:
            rule_string = '{:.4f}\t{:.4f}\t{:.4f}\tr_{}: '.format(rule.cov(), rule.conf(), rule.supp(), i)
            rule_string = rule_string + ' ˄ '.join(map(lambda cond: str(self._global_condition_map[cond]), rule._A))
            rule_string += ' → ' + str(rule._y)
            rule_string += '\n'
            return_str += rule_string
            i += 1
        print(return_str)

    def _initialize_sets(self):
        self._bad_combinations = set()
        self._good_combinations = set()
        self._cond_cov_dict = self._compute_condition_bit_sets()
        # self._training_bit_sets = self._compute_training_bit_sets()

    def _sort_ruleset(self, ruleset):
        if self.rule_order == 'cov':
            ruleset.sort(key=lambda rule: (rule.cov(), len(rule.A()) * -1, rule.conf()), reverse=True)
        elif self.rule_order == 'conf':
            ruleset.sort(key=lambda rule: (rule.conf(), len(rule.A()) * -1, rule.cov()), reverse=True)


    # return a dictionary for each class containing the bitsets for coverage that class
    def _compute_condition_bit_sets(self):
        # empty sets for each condition coverage class
        cond_cov_dict = [{} for _ in range(len(self.classes_))]
        for cond_id, cond in self._global_condition_map.items():
            # empty bitarray of 0's for each class
            cond_coverage_bitarray = [helpers.zero_bitarray(self.X_.shape[0]) for _ in range(len(self.classes_))]
            # traverse the training set, check if condition satisfies
            for row_idx, (x_row, y) in enumerate(zip(self.X_, self.y_)):
                att_idx = cond.attribute_index()
                if cond.satisfies(x_row[att_idx]):
                    # check the class y of the training set and set the bit to true for that instance
                    class_index = np.where(self.classes_ == y)[0].item()
                    cond_coverage_bitarray[class_index][row_idx] = True
            for class_idx in range(len(self.classes_)):
                # create the entries in the dictionary
                cond_cov_dict[class_idx][cond_id] = cond_coverage_bitarray[class_idx]
        return cond_cov_dict

    def _compute_training_bit_sets(self):
        training_set_bitarray = [helpers.zero_bitarray(self.X_.shape[0]) for _ in range(len(self.classes_))]
        # first create a list of bitarray for the actual values of training set
        for row_idx, (x_row, y) in enumerate(zip(self.X_, self.y_)):
            # check the class y of the training set and set the bit to true for that instance
            class_index = np.where(self.classes_ == y)[0].item()
            training_set_bitarray[class_index][row_idx] = True

        return training_set_bitarray

    # main guiding procedure for combining rulesets for classification, make a combination of each of the class with
    # itself and all the other classes
    def _combine_rulesets(self, ruleset1, ruleset2):
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if rule1.y() == [class_one]]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if rule2.y() == [class_two]]
                combined_rules.update(self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        return list(combined_rules)

    # actual combination procedure between to class-sliced rulesets
    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A()) == 0 or len(r2.A()) == 0:
                    continue
                self._n_combinations += 1  # count the actual number of combinations
                r1_AUr2_A = r1.A(frozen=False).union(r2.A(frozen=False))
                self._simplify_conditions(r1_AUr2_A)

                # create the new rule and compute class distribution and predicted class
                weight = None
                if self._base_ens_type == 'classifier':
                    if self._weights is None:
                        class_dist = np.mean([r1.class_distribution(), r2.class_distribution()],
                                             axis=0).reshape((len(self.classes_),))
                    else:
                        class_dist = np.average([r1.class_distribution(), r2.class_distribution()], axis=0,
                                                weights=[r1.weight(), r2.weight()]).reshape((len(self.classes_),))
                        weight = (r1.weight()+r2.weight) / 2
                    y_class_index = np.argmax(class_dist).item()
                    y = [self.classes_[y_class_index]]
                    logit_score = 0
                elif self._base_ens_type == 'gradient_classifier':
                    logit_score = r1.logit_score() + r2.logit_score()
                    if len(self.classes_) == 2:
                        raw_to_proba = expit(logit_score)
                        class_dist = [raw_to_proba.item(), 1 - raw_to_proba.item()]
                    else:
                        class_dist = logit_score - logsumexp(logit_score)
                    y_class_index = np.argmax(class_dist).item()
                    y = [self.classes_[y_class_index]]
                elif self._base_ens_type == 'regressor':
                    y = np.mean([r1.y(), r2.y()], axis=0)

                n_samples = np.sum([r1.n_samples(), r2.n_samples()], axis=0)
                new_rule = Rule(frozenset(r1_AUr2_A), class_dist=class_dist, logit_score=logit_score, y=y,
                                y_class_index=y_class_index, n_samples=n_samples, classes=self.classes_,weight=weight)

                new_rule_cov, new_rule_conf_supp = self._get_conditions_measures(r1_AUr2_A)
                new_rule.set_measures(new_rule_cov, new_rule_conf_supp[y_class_index][0],
                                      new_rule_conf_supp[y_class_index][1])

                # check if the combination was null before, if it was we just skip it
                if new_rule in self._bad_combinations:
                    continue
                # if the combination was a good one before, we just add the combination to the rules
                if new_rule in self._good_combinations:
                    combined_rules.add(new_rule)
                else:
                    if new_rule.cov() > self.cov_threshold and \
                            new_rule.cov() != 1 and \
                            new_rule.conf() > self.conf_threshold and \
                            len(new_rule.A()) <= self.max_rule_depth:
                        combined_rules.add(new_rule)
                        self._good_combinations.add(new_rule)
                    else:
                        self._bad_combinations.add(new_rule)

        return combined_rules

    # get the measures of individual conditions registered from the trees
    def _get_conditions_measures(self, conditions, mask=None, return_set_size=False):
        if len(conditions) == 0:
            return 0.0, np.zeros((self.classes_.shape[0], 2))
        cov_set_by_class = None  # store the list of coverage sets by class
        cov_set = None  # store the coverage set for that condition (it is the OR operation between class coverages)
        for cond in conditions:
            if cov_set_by_class is None:  # iteration 0
                cov_set_by_class = [cov[cond] for cov in self._cond_cov_dict]
                cov_set = helpers.list_or_operation(cov_set_by_class)
            else:
                current_cond_cov = [cov[cond] for cov in self._cond_cov_dict]
                cov_set_by_class = [cum_cov & curr_cov for cum_cov, curr_cov in zip(cov_set_by_class, current_cond_cov)]
                cov_set = helpers.list_or_operation(cov_set_by_class)
                if cov_set.count() == 0:
                    return 0.0, np.zeros((self.classes_.shape[0], 2))
        if mask is not None:
            cov_set_by_class = [cum_cov & mask for cum_cov in cov_set_by_class]
            cov_set = helpers.list_or_operation(cov_set_by_class)
            if cov_set.count() == 0:
                return 0.0, np.zeros((self.classes_.shape[0], 2))
            updated_mask = ~cov_set & mask
            mask.clear()
            mask.extend(updated_mask)
        coverage = cov_set.count() / self.X_.shape[0]
        if return_set_size:
            return cov_set.count(), [cov.count() for cov in cov_set_by_class]
        #  return cov(r) ,[conf(r), supp(r)] - list by class
        return coverage, [[cov.count() / cov_set.count(), cov.count() / self.X_.shape[0]] for cov in cov_set_by_class]

    # simplify the conditions of a set of conditions, removing redundant conditions
    def _simplify_conditions(self, conditions):
        cond_map = self._global_condition_map  # just for readability
        # create list with this format ["(att_index, 'OPERATOR')", 'cond_id']
        att_op_list = [[str((cond_map[cond].attribute_index(), cond_map[cond].operator().name)), cond]
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
            if tup_att_op[1] in ['LESS_THAN', 'LESS_OR_EQUAL_THAN']:
                edge_condition = max(list_conds, key=lambda item: item.value())  # condition at the edge of the box
            if tup_att_op[1] in ['GREATER_THAN', 'GREATER_OR_EQUAL_THAN']:
                edge_condition = min(list_conds, key=lambda item: item.value())
            list_conds.remove(edge_condition)  # remove the edge condition of the box from the list, so it will remain
            [conditions.remove(hash(cond)) for cond in list_conds]

        att_op_list = [[cond_map[cond].attribute_index(), cond_map[cond].operator().name, cond]
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
            list_conds = [(int(id_), self._get_conditions_measures({int(id_)})[0]) for id_ in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(best_condition)  # remove the edge condition of the box from the list so it will remain
            [conditions.remove(cond[0]) for cond in list_conds]

    def _compute_rule_measures(self, combined_rules, sequential_coverage=False):
        uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
        if sequential_coverage:
            combined_rules[:] = [rule for rule in combined_rules if self._rule_is_accurate(rule, uncovered_instances)]
        else:
            [rule for rule in combined_rules if self._rule_is_accurate(rule, uncovered_instances)]
        # print(uncovered_instances.count())
        # for rule in combined_rules:
        #     if uncovered_instances.count() < 50:
        #         stop=[]
        #     if not self._rule_is_accurate(rule, uncovered_instances):
        #         combined_rules.remove(rule)
        stop=[]

    def _rule_is_accurate(self, rule, uncovered_instances):
        if uncovered_instances.count() == 0:
            return False
        local_uncovered_instances = copy.copy(uncovered_instances)
        rule_cov, rule_conf_supp = self._get_conditions_measures(rule.A(), mask=local_uncovered_instances)
        rule_conf = rule_conf_supp[rule.class_index()][0]
        rule_supp = rule_conf_supp[rule.class_index()][1]
        rule.set_measures(rule_cov, rule_conf, rule_supp)
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
            base_line_error = self._compute_pessimistic_error(rule.A(), rule.class_index())
            min_error = 0
            while min_error < base_line_error and len(rule.A()) > 1:
                errors = [(cond, self._compute_pessimistic_error(rule.A().difference([cond]), rule.class_index()))
                          for cond in rule.A()]
                min_error_tup = min(errors, key=lambda tup: tup[1])
                min_error = min_error_tup[1]
                if min_error < base_line_error:
                    base_line_error = min_error
                    min_error = 0
                    rule_conds = rule.A(frozen=False)
                    rule_conds.remove(min_error_tup[0])
                    rule.set_A(rule_conds)

        combined_rules[:] = [rule for rule in combined_rules if len(rule.A()) > 0]
        # combined_rules.sort(key=lambda r: (r.cov(), r.conf()), reverse=True)
        self._sort_ruleset(combined_rules)
        self._compute_rule_measures(combined_rules, sequential_coverage=True)

    def _compute_pessimistic_error(self, conditions, class_index):
        cov, class_cov = self._get_conditions_measures(conditions, return_set_size=True)
        total_instances = cov
        accurate_instances = class_cov[class_index]

        error_instances = total_instances - accurate_instances
        alpha_half = 1.15  # 25 % confidence for C4.5
        e = error_instances / total_instances  # totalInstances

        return 100 * _pessimistic_error_rate(total_instances, e, alpha_half)

    def _add_default_rule(self, simplified_ruleset):
        uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
        [rule for rule in simplified_ruleset if self._rule_covers(rule, uncovered_instances)]

        all_covered = False
        if uncovered_instances.count() == 0:
            uncovered_instances = helpers.one_bitarray(self.X_.shape[0])
            all_covered = True

        uncovered_labels = self.y_[uncovered_instances.tolist()]
        uncovered_dist = np.array([(uncovered_labels == class_).sum() for class_ in self.classes_])
        default_class_idx = np.argmax(uncovered_dist)
        default_rule = Rule({}, class_dist=uncovered_dist / uncovered_dist.sum(),
                            y=[self.classes_[default_class_idx]], y_class_index=default_class_idx,
                            classes=self.classes_)
        if not all_covered:
            default_rule.set_measures(cov=uncovered_instances.count()/self.X_.shape[0],
                                      conf=uncovered_dist[default_class_idx]/uncovered_instances.count(),
                                      supp=uncovered_dist[default_class_idx]/self.X_.shape[0])
        simplified_ruleset.append(default_rule)
        return True

    def _rule_covers(self, rule, uncovered_instances):
        local_uncovered_instances = copy.copy(uncovered_instances)
        _, _ = self._get_conditions_measures(rule.A(), mask=local_uncovered_instances)
        uncovered_instances.clear()
        uncovered_instances.extend(local_uncovered_instances)

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        rule_added = self._add_default_rule(simplified_ruleset)
        sim_rules_perf = f1_score(self.y_,
                                  RuleSet(simplified_ruleset, self._global_condition_map).predict(self.X_))
        if rule_added:
            simplified_ruleset.pop()

        rule_added = self._add_default_rule(combined_rules)
        comb_rules_perf = f1_score(self.y_,
                                   RuleSet(combined_rules, self._global_condition_map).predict(self.X_))
        if rule_added:
            combined_rules.pop()

        if comb_rules_perf >= sim_rules_perf:
            return combined_rules
        else:
            return simplified_ruleset

    def _get_gbm_init(self):
        if isinstance(self._base_ens,GradientBoostingClassifier):
            return self._base_ens._raw_predict_init(self.X_[0].reshape(1, -1))
        if isinstance(self._base_ens, XGBClassifier):
            return self._base_ens.base_score
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

    def fit(self, X, y, sample_weight=None, column_names=None):
        super().fit(X, y, sample_weight=sample_weight, column_names=column_names)

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
        return self._base_ens

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
        return self._base_ens.predict(X)
