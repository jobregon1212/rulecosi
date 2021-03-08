""" Supporting classes for handling rulesets, rules and conditions for RuleCOSI

"""


import numpy as np
import pandas as pd
import operator
from functools import reduce

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.utils import check_array

op_dict = {'eq': '=',
           'gt': '>',
           'lt': '<',
           'ge': '≥',
           'le': '≤',
           'ne': '!='}


class RuleSet:
    """ A set of ordered rules that can be used to make predictions.

    Parameters
    ----------
    rules : array of Rules, default=None
        Rules belonging to the ruleset

    condition_map: dictionary of <condition_id, Condition>, default=None
    Dictionary of Conditions used in the ruleset. condition_id is an integer
    uniquely identifying the Condition.

    ruleset: Ruleset
        If different than None, copy that ruleset properties into this object


    """
    def __init__(self, rules=None, condition_map=None, ruleset=None):
        if condition_map is None:
            condition_map = {}
        if rules is None:
            rules = []
        if ruleset is None:
            self.rules = rules
            self.condition_map = condition_map
            self.n_total_ant = 0
            self.n_uniq_ant = 0
        else:
            self.rules = ruleset.rules
            self.condition_map = ruleset.condition_map
            self.n_total_ant = ruleset.n_total_ant
            self.n_uniq_ant = ruleset.n_uniq_ant
        self.geometric_mean_score = 0
        self.f1_score = 0
        self.accuracy_score = 0
        self.roc_auc = 0

    def prune_condition_map(self):
        """Prune the condition map in this ruleset to contain only the
        conditions present in the rules

        """
        condition_set = {cond
                         for rule in self.rules
                         for cond in rule.A}
        self.condition_map = {key: self.condition_map[key] for key in condition_set}

    def __str__(self):
        return_str = ''
        i = 1
        for rule in self.rules:
            rule_string = 'r_{}: '.format(i)
            rule_string = rule_string + ' ˄ '.join(map(lambda cond: str(self.condition_map[cond]), rule.A))
            rule_string += ' → ' + str(rule.y)
            rule_string += '\n'
            return_str += rule_string
            i += 1
        return return_str

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.rules)

    def compute_interpretability_measures(self):
        """Compute the following interpretability measures:

        - n_rules: number of rules
        - n_uniq_ant: number of unique antecedents or conditions
        - n_total_ant: total number of antecedents or conditions

        """
        self.prune_condition_map()
        n_uniq_ant = len(self.condition_map)
        n_total_ant = 0
        for rule in self.rules:
            n_total_ant += len(rule.A)
        return len(self.rules), n_uniq_ant, n_total_ant

    def compute_classification_performance(self, X, y_true, metric='gmean'):
        """ Compute the classification performance measures of this RuleSet

        :param  X : array-like, shape (n_samples, n_features)
            The input samples.
        :param y_true: array-like, shape (n_samples,)
            The real target value
        :param metric: string, default='gmean'
           Metric that is computed for this RuleSet. Other accepted measures are:
         - 'f1' for F-measure
         - 'roc_auc' for AUC under the ROC curve
         - 'accuracy' for Accuracy

        """
        if metric == 'roc_auc':
            y_score = self.predict_proba(X)
            self.roc_auc = roc_auc_score(y_true, y_score[:, 1])
        else:
            y_pred = self.predict(X)
            if metric == 'gmean':
                self.geometric_mean_score = geometric_mean_score(y_true, y_pred)
            elif metric == 'f1':
                self.f1_score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                self.accuracy_score = accuracy_score(y_true, y_pred)
            else:
                self.geometric_mean_score = geometric_mean_score(y_true, y_pred)

    def metric(self, metric='gmean'):
        """ Return the metric value of this RuleSet

        :param metric: string, default='gmean'
            Other accepted measures are:
             - 'f1' for F-measure
             - 'roc_auc' for AUC under the ROC curve
             - 'accuracy' for Accuracy

        """
        if metric == 'gmean':
            return self.geometric_mean_score
        elif metric == 'f1':
            return self.f1_score
        elif metric == 'roc_auc':
            return self.roc_auc
        elif metric == 'accuracy':
            return self.accuracy_score
        else:
            return self.geometric_mean_score

    def compute_all_classification_performance(self, X, y_true):
        """ Compute all the classification performance measures of this RuleSet

        :param  X : array-like, shape (n_samples, n_features)
            The input samples.
        :param y_true: array-like, shape (n_samples,)
            The real target value

        """
        X = check_array(X)
        # y_true = check_array(y_true.reshape(1, -1))
        y_pred = self.predict(X)
        y_score = self.predict_proba(X)
        self.geometric_mean_score = geometric_mean_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.accuracy_score = accuracy_score(y_true, y_pred)
        self.roc_auc = roc_auc_score(y_true, y_score[:, 1])

    def predict(self, X):
        """ Make predictions for the input values in X.

        :param X: X : array-like, shape (n_samples, n_features)
            The input samples.
        :return: y_pred: array-like, shape (n_samples,)
            The predicted target values
        """
        return np.ravel(self._predict(X)[0])

    def predict_proba(self, X):
        """ Make probability predictions for the input values in X.

         :param X: X : array-like, shape (n_samples, n_features)
             The input samples.

         :return:  ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.

         """
        return self._predict(X, proba=True)[0]

    def _predict(self, X, proba=False):
        """ Make predictions for the input values in X.

        :param X: X : array-like, shape (n_samples, n_features)
            The input samples.

        :param proba: boolean, default=False Determines if the predictions
        are the target values or target probability values.

        :return: (y_pred, covered_mask): tuple containing:
                 0 - array-like, shape (n_samples,) or  ndarray of shape
                 (n_samples, n_classes)

                 1 - array of booleans containing the mask of covered masks by
                 this RuleSet

            The predicted target values or target probabilities

        """
        if proba:
            prediction = np.empty((X.shape[0], self.rules[0].class_dist.shape[0]))
        else:
            prediction = np.empty((X.shape[0], self.rules[0].y.shape[0]), dtype=self.rules[0].y.dtype)

        covered_mask = np.zeros((X.shape[0],), dtype=bool)  # records the records that are already covered by some rule
        for i, rule in enumerate(self.rules):
            r_pred, r_mask = rule.predict(X, condition_map=self.condition_map, proba=proba)
            # update the covered_mask with the records covered by this rule
            remaining_covered_mask = ~covered_mask & r_mask
            # and then update the predictions of the remaining uncovered cases with the covered records of this rule
            prediction[remaining_covered_mask] = r_pred[remaining_covered_mask]
            covered_mask = covered_mask | r_mask
            if covered_mask.sum() == X.shape[0]:
                break

        return prediction, covered_mask

    def print_rules(self, return_object=None, heuristics_digits=4,
                    condition_digits=3):
        """ Print the rules in a string format. It can also return an object
        containing the rules and its heuristics

        :param return_object: string, default=None
            Indicates if the rules should be returned in an object. Possible
            values are:
            - 'string': it returns a string containing the rules in a readable
               format
            - 'dataframe': returns a :class:`pandas.DataFrame` object containing
               the rules

        :param heuristics_digits: number of decimal digits to be displayed in the
        heuristics of the rules

        :param condition_digits: number of decimal digits to be displayed in the
        conditions of the rules

        :return: str or :class:`pandas.DataFrame`
        """

        return_str = 'cov \tconf \tsupp \tsamples \t\trule\n'
        columns = ['cov', 'conf', 'supp', 'samples', '#', 'A', 'y']
        rule_rows = []
        i = 1
        for rule in self:
            samples = ','.join(rule.n_samples.astype(str))
            samples = f'[{samples}]'
            if len(samples) > 7:
                sample_tab = '\t'
            else:
                sample_tab = '\t\t'
            rule_string = f'{rule.cov:.{heuristics_digits}f}\t{rule.conf:.{heuristics_digits}f}\t{rule.supp:.{heuristics_digits}f}\t{samples}{sample_tab}r_{i}: '
            rule_row = [f'{rule.cov:.{heuristics_digits}f}', f'{rule.conf:.{heuristics_digits}f}', f'{rule.supp:.{heuristics_digits}f}', samples, f'r_{i}']
            if len(rule.A) == 0:
                rule_string = rule_string + '( )'
                rule_row.append('()')
            else:
                A_string = ' ˄ '.join(
                    map(lambda cond: self.condition_map[cond].__str__(digits=condition_digits), rule.A))
                rule_string = rule_string + A_string
                rule_row.append(A_string)
            rule_string += ' → ' + str(rule.y)
            rule_row.append(rule.y)
            rule_string += '\n'
            return_str += rule_string
            i += 1
            rule_rows.append(rule_row)
        if return_object is not None:
            if return_object == 'string':
                return return_str
            elif return_object == 'dataframe':
                return pd.DataFrame(data=rule_rows, columns=columns)
        else:
            print(return_str)


class Rule:
    """ Represents a single rule wich has the form r: A -> y.

        A is a set of conditions also called body of the rule.
        y is the predicted class, also called head of the rule.
    """

    def __init__(self, conditions, class_dist=None, logit_score=None, y=None, y_class_index=None, n_samples=None,
                 n_outputs=1, classes=None, weight=0):
        if classes is None:
            self.classes = [0, 1]
        else:
            self.classes = list(classes)
        if class_dist is None:
            class_dist = np.array([1.0 / len(self.classes) for _ in self.classes])
        if n_samples is None:
            self.n_samples = np.array([0 for _ in self.classes])

        self.A = conditions  # conditions
        self.class_dist = class_dist
        self.logit_score = logit_score
        self.y = y
        self.class_index = y_class_index
        self.n_outputs = n_outputs
        self.cov = 0
        self.supp = 0
        self.conf = 0
        self.weight = weight

    def __str__(self):
        if len(self.A) == 0:
            return_string = '( )'
        else:
            return_string = ' ˄ '.join(map(str, self.A))
        return_string += ' → ' + str(self.y)
        return return_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Rule):
            if id(self) == id(other):
                return True
            if self.A != other.A:
                return False
            return self.y == other.y
        return False

    def __hash__(self):
        return hash((self.A, tuple(self.y)))

    def predict(self, X, condition_map, proba=False):
        """ Make predictions for the input values in X.

        :param X: X : array-like, shape (n_samples, n_features)
            The input samples.

        :param condition_map: dictionary of <condition_id, Condition>
        Dictionary of Conditions used in the ruleset. condition_id is an
        integer uniquely identifying the Condition.

        :param proba: boolean, default=False Determines if the predictions
        are the target values or target probability values.


        :return: (y_pred, covered_mask): tuple containing:
                 0 - array-like, shape (n_samples,) or  ndarray of shape
                   (n_samples, n_classes)
                 1 - array of booleans containing the mask of covered masks by
                    this RuleSet

        """
        mask = np.ones((X.shape[0]), dtype=bool)

        # apply the and operator to all "satisfies _array" functions of the conditions of this rule
        if len(self.A) > 0:
            mask = reduce(operator.and_, [condition_map[cond].satisfies_array(X) for cond in self.A])
        if proba:
            prediction = np.zeros((X.shape[0], self.class_dist.shape[0]))
            prediction[mask] = self.class_dist
        else:
            prediction = np.zeros((X.shape[0], self.y.shape[0]))
            prediction[mask] = self.y
        return prediction, mask

    def get_condition(self, att_index):
        """ Returns the condition belonging to the att_index

        :param att_index: int
            The attribute index that wants to be retrieved.

        :return: If the attribute exists, returns the attribute, otherwise
        returns None
        """
        for cond in self.A:
            if cond.get_attribute_index() == att_index:
                return cond
        return None

    def add_condition(self, condition):
        """ Add a condition to the rule

        :param condition: Condition to be added

        """
        if condition not in self.A:
            self.A.add(condition)

    def set_heuristics(self, heuristics_dict):
        """ Set the heuristics of this rule contained in the heuristics
        dictionary.

        :param heuristics_dict: A dictionary containing the heuristics values
        for this Rule

        """
        self.conf = heuristics_dict['conf'][self.class_index]
        self.supp = heuristics_dict['supp'][self.class_index]
        self.cov = heuristics_dict['cov']
        if self.cov > 0:
            self.n_samples = np.array([heuristics_dict['cov_set'][i].count() for i in range(len(self.classes))])


class Condition:
    """ Class representing a Rule Condition.

    A condition is an evaluation of an operator with a value. The operator
    could be any of the ones contained in op_dict = {'eq': '=', 'gt': '>',
    'lt': '<', 'ge': '≥', 'le': '≤', 'ne': '!='}

    """

    def __init__(self, att_index, op, value, att_name=None):
        self.att_index = int(att_index)
        if att_name is None:
            self.att_name = 'att_' + str(att_index)
        else:
            self.att_name = att_name
        self.op = op
        self.value = float(value)

    def __str__(self, digits=3):
        # return '({} {} {:.3f})'.format(self.att_name, op_dict[self.op.__name__], self.value)
        return f'({self.att_name} {op_dict[self.op.__name__]} {self.value:.{digits}f})'


    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Condition):
            if hash(self) == hash(other):
                return True
            if self.att_index != other.att_index:
                return False
            if self.op != other.op:
                return False
            return self.value == other.value
        return False

    def __hash__(self):
        return hash((self.att_index, self.op, self.value))

    def satisfies(self, value):
        """ Evaluates if the condition is satisfied or not using the provided
        value.

        :param value: int
            The value used for evaluating the condition

        :return: boolean
             True if the condition is satisfied and False otherwise
        """
        return self.op(value, self.value)

    def satisfies_array(self, arr):
        """ Evaluates if the condition is satisfied or not for all the
        records in the provided array.

        It applies the operator to the values in arr of the column equal to
        the index of the attribute of this condition and returns an array of
        bool

        :param arr: array-like, shape (n_samples, n_features)
            The input samples.

        :return: array of booleans Array of booleans denoting if the
        condition was satisfied on each of the elements of arr
        """
        return self.op(arr[:, self.att_index], self.value)