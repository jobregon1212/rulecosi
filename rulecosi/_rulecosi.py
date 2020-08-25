"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import copy
from bitarray import bitarray, frozenbitarray
from bitarray.util import count_and
from ast import literal_eval
from rulecosi.rules import Condition, Rule, Operator, RuleSet
from scipy.special import expit, logsumexp
# from rulecosi.helpers import remove_duplicated_rules,order_trees_by_weight
import rulecosi.helpers as helpers


def _ensemble_type(ensemble):
    if isinstance(ensemble, (AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)):
        return 'classifier'
    elif isinstance(ensemble, GradientBoostingClassifier):
        return 'gradient_classifier'
    else:
        return 'regressor'


def _ensemble_has_weights(ensemble):
    if isinstance(ensemble, (AdaBoostClassifier, AdaBoostRegressor)):
        return True
    else:
        return False


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
                 random_state=None):
        self._base_ens = base_ensemble
        self._base_ens_type = None
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.classes_ = None
        self._weights = None

        self.X_ = None
        self.y_ = None

        self._original_rulesets = None
        self.simplified_ruleset = None

        self._global_condition_map = None
        self._cond_cov_dict = None

        self._null_combinations = None
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
                column_names = X.columns

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
                self._base_ens = AdaBoostRegressor(self.base_estimator,
                                                   n_estimators=self.n_estimators,
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
        self._original_rulesets, self._global_condition_map = self._extract_rulesets(self._base_ens,
                                                                                     base_ensemble_type=self._base_ens_type,
                                                                                     column_names=column_names)
        processed_rulesets = copy.deepcopy(self._original_rulesets)
        # If the ensemble has weights that are different for each tree, the trees are ordered according their weights
        if _ensemble_has_weights(self._base_ens):
            self._weights = copy.copy(self._base_ens.estimator_weights_)
            # check if weights are the same
            if np.max(self._weights) != np.min(self._weights):
                # order the rulesets according weight
                self._weights, processed_rulesets = helpers.order_trees_by_weight(processed_rulesets,
                                                                                  self._weights)

        # Then the duplicated rules are removed
        processed_rulesets, self._weights, _ = helpers.remove_duplicated_rules(processed_rulesets,
                                                                               self._weights)
        self.simplified_ruleset = processed_rulesets[0]

        self._initialize_sets()

        self._n_combinations = 0

        for i in range(1, len(processed_rulesets)):
            combined_rules = self._combine_rulesets(self.simplified_ruleset, processed_rulesets[i])

            # order by rule confidence

            # prune innacurate rules

            #optimize rules

            test = RuleSet(list(combined_rules),self._global_condition_map)
            stop = ['stop']

    def _initialize_sets(self):
        self._null_combinations = set()
        self._good_combinations = set()
        self._cond_cov_dict = self._compute_condition_sets()

    def _extract_rulesets(self, original_ensemble, base_ensemble_type, column_names):
        rulesets = []
        global_condition_map = dict()
        if self._base_ens_type == 'classifier':
            for base_tree in original_ensemble:
                original_ruleset = self.get_base_ruleset(base_tree, column_names=column_names,
                                                         base_ensemble_type=base_ensemble_type)
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.get_condition_map())
        elif self._base_ens_type == 'gradient_classifier':
            for tree_index, base_trees in enumerate(original_ensemble):
                for class_index, base_tree in enumerate(base_trees):
                    original_ruleset = self.get_base_ruleset(base_tree, column_names=column_names,
                                                             base_ensemble_type=base_ensemble_type,
                                                             class_index=class_index, tree_index=tree_index)
                    rulesets.append(original_ruleset)
                    global_condition_map.update(original_ruleset.get_condition_map())
        return rulesets, global_condition_map

    def get_base_ruleset(self, base_tree, column_names=None, condition_map=None, base_ensemble_type='classifier',
                         class_index=None, tree_index=None):
        check_is_fitted(self, ['X_', 'y_'])
        tree = base_tree.tree_
        # n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        n_samples = tree.weighted_n_node_samples

        if condition_map is None:
            condition_map = dict()  # dictionary of conditions A

        def _recursive(node_index, condition_set):
            rules = []
            # leaf node so a rule is created
            if children_left[node_index] == children_right[node_index]:
                logit_score = None
                if base_ensemble_type == 'classifier':
                    class_dist = (value[node_index] / value[node_index].sum())
                    y_class_index = np.argmax(class_dist, axis=1).item()
                    y = [self.classes_[y_class_index]]
                elif base_ensemble_type == 'gradient_classifier':
                    if tree_index == 0:
                        init = self._base_ens._raw_predict_init(self.X_[0].reshape(1, -1))
                    else:
                        init = np.zeros(value[node_index].shape)
                    logit_score = init + value[node_index]
                    raw_to_proba = expit(logit_score)
                    if len(self.classes_) == 2:
                        class_dist = [[raw_to_proba, 1 - raw_to_proba]]
                    else:
                        class_dist = logit_score - logsumexp(logit_score)
                    y_class_index = np.argmax(class_dist, axis=1).item()
                    y = [self.classes_[y_class_index]]
                elif base_ensemble_type == 'regressor':
                    y = value
                new_rule = Rule(frozenset(condition_set), class_dist=class_dist, logit_score=logit_score, y=y,
                                y_class_index=y_class_index, n_samples=n_samples[node_index], classes=self.classes_)
                rules.append(new_rule)
            else:
                att_name = None
                if column_names is not None:
                    att_name = column_names[feature[node_index]]
                condition_set_left = copy.deepcopy(condition_set)
                new_condition_left = Condition(feature[node_index], Operator.LESS_OR_EQUAL_THAN, threshold[node_index],
                                               att_name)
                # print(new_condition_left)
                condition_map[hash(new_condition_left)] = new_condition_left
                condition_set_left.add(hash(new_condition_left))
                left_rules = _recursive(children_left[node_index], condition_set_left)
                rules = rules + left_rules

                condition_set_right = copy.deepcopy(condition_set)
                new_condition_right = Condition(feature[node_index], Operator.GREATER_THAN, threshold[node_index],
                                                att_name)
                # print(new_condition_right)
                condition_map[hash(new_condition_right)] = new_condition_right
                condition_set_right.add(hash(new_condition_right))
                right_rules = _recursive(children_right[node_index], condition_set_right)
                rules = rules + right_rules
            return rules

        empty_condition_set = set()
        return_rules = _recursive(0, empty_condition_set)
        return RuleSet(return_rules, condition_map)

    # return a dictionary for each class containing the bitsets for coverage that class
    def _compute_condition_sets(self):
        cond_cov_dict = [{} for _ in range(len(self.classes_))]
        for cond_id, cond in self._global_condition_map.items():
            cond_coverage_bitarray = [helpers.zero_bitarray(self.X_.shape[0]) for _ in range(len(self.classes_))]
            for row_idx, (x_row, y) in enumerate(zip(self.X_, self.y_)):
                att_idx = cond.attribute_index()
                if cond.satisfies(x_row[att_idx]):
                    class_index = np.where(self.classes_ == y)[0].item()
                    cond_coverage_bitarray[class_index][row_idx] = True
            for class_idx in range(len(self.classes_)):
                cond_cov_dict[class_idx][cond_id] = cond_coverage_bitarray[class_idx]
        return cond_cov_dict

    def _combine_rulesets(self, ruleset1, ruleset2):
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if rule1.y() == [class_one]]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if rule2.y() == [class_two]]
                combined_rules.update(self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        return combined_rules

    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                self._n_combinations+=1
                r1_AUr2_A = r1.A(frozen=False).union(r2.A(frozen=False))
                self._simplify_conditions(r1_AUr2_A)

                if r1_AUr2_A in self._null_combinations:
                    continue

                if r1_AUr2_A in self._good_combinations:
                    combined_rules.add()
                else:
                    if self._base_ens_type == 'classifier':
                        class_dist = np.mean([r1.class_distribution(), r2.class_distribution()], axis=0)
                        y_class_index = np.argmax(class_dist, axis=1).item()
                        y = [self.classes_[y_class_index]]
                    elif self._base_ens_type == 'gradient_classifier':
                        logit_score = r1.logit_score() + r2.logit_score()
                        if len(self.classes_) == 2:
                            raw_to_proba = expit(logit_score)
                            class_dist = [[raw_to_proba, 1 - raw_to_proba]]
                        else:
                            class_dist = logit_score - logsumexp(logit_score)
                        y_class_index = np.argmax(class_dist, axis=1).item()
                        y = [self.classes_[y_class_index]]
                    elif self._base_ens_type == 'regressor':
                        y = np.mean([r1.y(), r2.y()], axis=0)

                    n_samples = np.sum([r1.n_samples(), r2.n_samples()],axis=0)
                    new_rule = Rule(frozenset(r1_AUr2_A), class_dist=class_dist, logit_score=logit_score, y=y,
                                    y_class_index=y_class_index, n_samples=n_samples, classes=self.classes_)

                    new_rule_cov, new_rule_conf_supp = self._get_conditions_measures(r1_AUr2_A)
                    new_rule.set_measures(new_rule_cov,new_rule_conf_supp[y_class_index][0],new_rule_conf_supp[y_class_index][1] )
                    if new_rule.cov() > 0 and new_rule.conf() > 0.5:
                        combined_rules.add(new_rule)
                        self._good_combinations.add(new_rule)
                    else:
                        self._null_combinations.add(new_rule)

        return combined_rules

    def _get_conditions_measures(self, conditions):
        class_n_coverage = None
        n_coverage = None
        for cond in conditions:
            if class_n_coverage is None:
                class_n_coverage = [cov[cond] for cov in self._cond_cov_dict]
                n_coverage = helpers.list_or_operation(class_n_coverage)
            else:
                cond_coverages = [cov[cond] for cov in self._cond_cov_dict]
                class_n_coverage = [cov1 & cov2 for cov1, cov2 in zip(class_n_coverage, cond_coverages)]
                n_coverage = helpers.list_or_operation(class_n_coverage)
                if n_coverage.count() == 0:
                    return 0.0, np.zeros((self.classes_.shape[0], 2))
        coverage = n_coverage.count() / self.X_.shape[0]
        return coverage, [[cov.count() / n_coverage.count(), cov.count()/self.X_.shape[0]] for cov in class_n_coverage]

    def _simplify_conditions(self, conditions):
        cond_map = self._global_condition_map  # just for readability
        # create list with this format ["(att_index, 'OPERATOR')", 'cond_id']
        att_op_list = [[str((cond_map[cond].attribute_index(), cond_map[cond].operator().name)), cond]
                       for cond in conditions]
        att_op_list = np.array(att_op_list)
        # create list of unique values and count
        dict_unique_values = {str(i[0]): att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 1]
                              for i in att_op_list}
        # create generator to traverse just conditions with the same att_index and operator that appear more than once
        generator = ((att_op, conditions) for (att_op, conditions) in dict_unique_values.items() if len(conditions) > 1)

        for (att_op, conds) in generator:
            tup_att_op = literal_eval(att_op)
            list_conds = {cond_map[int(id_)] for id_ in conds}
            if tup_att_op[1] in ['LESS_THAN', 'LESS_OR_EQUAL_THAN']:
                edge_condition = max(list_conds, key=lambda item: item.value())  # condition at the edge of the box
            if tup_att_op[1] in ['GREATER_THAN', 'GREATER_OR_EQUAL_THAN']:
                edge_condition = min(list_conds, key=lambda item: item.value())
            [conditions.remove(hash(cond)) for cond in list_conds]

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
        super().fit(X, y, sample_weight)

        return self._base_ens

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

        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self._base_ens.predict(X)


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
