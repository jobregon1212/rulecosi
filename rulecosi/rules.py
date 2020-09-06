import numpy as np
import operator
from functools import reduce

from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score


class RuleSet:

    def __init__(self, ruleset):
        self._rules = ruleset.get_rule_list()
        self._condition_map = ruleset.get_condition_map()
        self.n_total_ant = ruleset.n_total_ant
        self.n_uniq_ant = ruleset.n_uniq_ant
        self.geometric_mean_score = 0
        self.f1_score = 0
        self.accuracy_score = 0

    def __init__(self, rules=[], condition_map={}):
        self._rules = rules
        self._condition_map = condition_map
        self.n_total_ant = 0
        self.n_uniq_ant = 0
        self.geometric_mean_score = 0
        self.f1_score = 0
        self.accuracy_score = 0

    def set_condition_map(self, condition_map):
        self._condition_map = condition_map

    def prune_condition_map(self):
        condition_set = {cond
                         for rule in self._rules
                         for cond in rule.A()}
        self._condition_map = {key: self._condition_map[key] for key in condition_set}

    def get_condition_map(self):
        return self._condition_map

    def get_rule(self, idx=0):
        return self._rules[idx]

    def get_rule_list(self):
        return self._rules

    def append_rule(self, rule):
        self._rules.append(rule)

    def append_rules(self, rules):
        self._rules = self._rules + rules

    def __str__(self):
        return_str = ''
        i = 1
        for rule in self._rules:
            # rule_string = ' ˄ '.join(map(lambda cond: str(self._A_star[cond]),self._A_star))
            rule_string = 'r_{}: '.format(i)
            rule_string = rule_string + ' ˄ '.join(map(lambda cond: str(self._condition_map[cond]), rule._A))
            rule_string += ' → ' + str(rule._y)
            rule_string += '\n'
            return_str += rule_string
            i += 1
        return return_str

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._rules)

    def compute_interpretability_measures(self):
        self.prune_condition_map()
        n_uniq_ant = len(self._condition_map)
        n_total_ant = 0
        for rule in self._rules:
            n_total_ant += len(rule.A())
        return len(self._rules), n_uniq_ant, n_total_ant

    def compute_classification_performance(self, X, y_true):
        y_pred = self.predict(X)
        self.geometric_mean_score = geometric_mean_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.accuracy_score = accuracy_score(y_true, y_pred)

    def predict(self, X):
        return self._predict(X)[0]

    def predict_proba(self, X):
        return self._predict(X, proba=True)[0]

    def _predict(self, X, proba=False):
        if proba:
            prediction = np.zeros((X.shape[0], self._rules[0].class_distribution().shape[0]))
        else:
            prediction = np.zeros((X.shape[0], self._rules[0].y().shape[0]))

        covered_mask = np.zeros((X.shape[0],), dtype=bool)  # records the records that are already covered by some rule
        for i, rule in enumerate(self._rules):
            r_pred, r_mask = rule.predict(X, condition_map=self._condition_map, proba=proba)
            # update the covered_mask with the records covered by this rule
            remaining_covered_mask = ~covered_mask & r_mask
            # and then update the predictions of the remaining uncovered cases with the covered records of this rule
            prediction[remaining_covered_mask] = r_pred[remaining_covered_mask]
            covered_mask = covered_mask | r_mask

        return prediction, covered_mask


class Rule:

    # def covers(self, X):
    #     if len(self._A) == 0:  # default rule
    #         return True
    #     return reduce(operator.and_, [cond.satisfies_array(X) for cond in self._A])

    def predict(self, X, condition_map, proba=False):
        if len(self._A) == 0:  # default rule
            # if there is no conditions in the rule, all the records satisfy and
            # an array of predicted class or class distribution for this rule is returned
            if proba:
                return np.tile(self._class_dist, (X.shape[0], self._class_dist.shape[0])), \
                       np.ones((X.shape[0]), dtype=bool)
            else:
                return np.tile(self._y, (X.shape[0], self._y.shape[0])), \
                       np.ones((X.shape[0]), dtype=bool)
        # apply the and operator to all satifies_array functions of the conditions of this rule
        mask = reduce(operator.and_, [condition_map[cond].satisfies_array(X) for cond in self._A])
        if proba:
            prediction = np.zeros((X.shape[0],self._class_dist.shape[0]))
            prediction[mask] = self._class_dist
        else:
            prediction = np.zeros((X.shape[0], self._y.shape[0]))
            prediction[mask] = self._y
        return prediction, mask

    def get_condition(self, att_index):
        for cond in self._A:
            if cond.get_attribute_index() == att_index:
                return cond
        return None

    def A(self, frozen=True):
        if frozen:
            return self._A
        else:
            return set(self._A)

    def set_A(self, conditions):
        self._A = frozenset(conditions)

    def y(self):
        return self._y

    def class_index(self):
        return self._class_index

    def class_distribution(self):
        return self._class_dist

    def logit_score(self):
        return self._logit_score

    def n_samples(self):
        return self._n_samples

    def cov(self):
        return self._cov

    def supp(self):
        return self._supp

    def conf(self):
        return self._conf

    def weight(self):
        return self._weight

    def add_condition(self, condition):
        if condition not in self._A:
            self._A.add(condition)

    def set_measures(self, cov, conf, supp):
        self._cov = cov
        self._conf = conf
        self._supp = supp

    def __init__(self, conditions, class_dist=None, logit_score=None, y=None, y_class_index=None, n_samples=0,
                 n_outputs=1, classes=None, weight=0):
        if class_dist is None:
            class_dist = np.array([0.5, 0.5])
        self._A = conditions  # conditions
        self._class_dist = class_dist
        self._logit_score = logit_score
        self._y = y
        self._class_index = y_class_index
        self._n_samples = n_samples
        self._n_outputs = n_outputs
        self._classes = list(classes)
        self._cov = 0
        self._supp = 0
        self._conf = 0
        self._weight = weight

    def __str__(self):
        if len(self._A) == 0:
            return_string = '( )'
        else:
            return_string = ' ˄ '.join(map(str, self._A))
        return_string += ' → ' + str(self._y)
        return return_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Rule):
            if id(self) == id(other):
                return True
            if self._A != other._A:
                return False
            return self._y == other._y
        return False

    def __hash__(self):
        return hash((self._A, tuple(self._y)))


# class Operator(Enum):
#     EQUAL_TO = '='
#     GREATER_THAN = '>'
#     LESS_THAN = '<'
#     GREATER_OR_EQUAL_THAN = '≥'
#     LESS_OR_EQUAL_THAN = '≤'
#     DIFFERENT_THAN = '!='

op_dict = {'eq': '=',
           'gt': '>',
           'lt': '<',
           'ge': '≥',
           'le': '≤',
           'ne': '!='}


class Condition:
    def satisfies(self, value):
        return self._op(value, self._value)

    def satisfies_array(self, arr):
        # apply the operator to the values in arr of the column equal to the index of the attribute of this condition
        # and returns an array of bool
        return self._op(arr[:, self._att_index], self._value)

    def attribute_index(self):
        return self._att_index

    def operator(self):
        return self._op

    def value(self):
        return self._value

    def __init__(self, att_index, _op, _value, att_name=None):
        self._att_index = int(att_index)
        if att_name is None:
            self._att_name = 'att_' + str(att_index)
        else:
            self._att_name = att_name
        self._op = _op
        self._value = float(_value)

    def __str__(self):
        return '({} {} {:.3f})'.format(self._att_name, op_dict[self._op.__name__], self._value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Condition):
            if hash(self) == hash(other):
                return True
            if self._att_index != other._att_index:
                return False
            if self._op != other._op:
                return False
            return self._value == other._value
        return False

    def __hash__(self):
        return hash((self._att_index, self._op, self._value))

