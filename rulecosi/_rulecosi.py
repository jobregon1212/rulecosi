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
from bitarray import bitarray,frozenbitarray
from bitarray.util import count_and
from rulecosi.rules import Condition, Rule, Operator, RuleSet
#from rulecosi.helpers import remove_duplicated_rules,order_trees_by_weight
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
        self._processed_rulesets = None
        self.simplified_ruleset = None

        self._global_condition_map = None
        self._cond_cov_dict= None

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

        self._base_ens_type = _ensemble_type(self._base_ens)

        # First step is extract the rules
        self._original_rulesets, self._global_condition_map = self._extract_rulesets(self._base_ens,
                                                                                     base_ensemble_type=self._base_ens_type,
                                                                                     column_names=column_names)

        # If the ensemble has weights that are different for each tree, the trees are ordered according their weights
        if _ensemble_has_weights(self._base_ens):
            self._weights = copy.copy(self._base_ens.estimator_weights_)
            # check if weights are the same
            if np.max(self._weights) != np.min(self._weights):
                # order the rulesets according weight
                self._weights, self._processed_rulesets = helpers.order_trees_by_weight(self._original_rulesets,
                                                                                        self._weights,)
        # Then the duplicated rules are removed
        self._processed_rulesets, self._weights, _ = helpers.remove_duplicated_rules(self._processed_rulesets,
                                                                                     self._weights)
        self.simplified_ruleset = self._processed_rulesets[0]

        self._cond_cov_dict = self._compute_condition_sets()

        stop = ['stop']

        #

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
            for base_trees in original_ensemble:
                for base_tree in base_trees:
                    original_ruleset = self.get_base_ruleset(base_tree, column_names=column_names,
                                                             base_ensemble_type=base_ensemble_type)
                    rulesets.append(original_ruleset)
                    global_condition_map.update(original_ruleset.get_condition_map())
        return rulesets, global_condition_map

    def get_base_ruleset(self, base_tree, column_names=None, condition_map=None, base_ensemble_type='classifier'):
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
            if children_left[node_index] == children_right[node_index]:
                new_rule = Rule(frozenset(condition_set), value=value[node_index], n_samples=n_samples[node_index],
                                type_=base_ensemble_type, classes=self.classes_)
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
                att_idx = cond.get_attribute_index()
                if cond.satisfies(x_row[att_idx]):
                    class_index = np.where(self.classes_ == y)[0].item()
                    cond_coverage_bitarray[class_index][row_idx] = True
            for class_idx in range(len(self.classes_)):
                cond_cov_dict[class_idx][cond_id] = cond_coverage_bitarray[class_idx]
        return cond_cov_dict

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
