import numpy as np
import operator
from functools import reduce

from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score


class RuleSet:

    def __init__(self, ruleset):
        self.rules = ruleset.get_rule_list()
        self.condition_map = ruleset.get_condition_map()
        self.n_total_ant = ruleset.n_total_ant
        self.n_uniq_ant = ruleset.n_uniq_ant
        self.geometric_mean_score = 0
        self.f1_score = 0
        self.accuracy_score = 0

    def __init__(self, rules=[], condition_map={}):
        self.rules = rules
        self.condition_map = condition_map
        self.n_total_ant = 0
        self.n_uniq_ant = 0
        self.geometric_mean_score = 0
        self.f1_score = 0
        self.accuracy_score = 0

    # def set_condition_map(self, condition_map):
    #     self.condition_map = condition_map

    def prune_condition_map(self):
        condition_set = {cond
                         for rule in self.rules
                         for cond in rule.A()}
        self.condition_map = {key: self.condition_map[key] for key in condition_set}

    # def get_condition_map(self):
    #     return self.condition_map
    #
    # def get_rule(self, idx=0):
    #     return self.rules[idx]
    #
    # def get_rule_list(self):
    #     return self.rules
    #
    # def append_rule(self, rule):
    #     self.rules.append(rule)
    #
    # def append_rules(self, rules):
    #     self.rules = self.rules + rules

    def __str__(self):
        return_str = ''
        i = 1
        for rule in self.rules:
            # rule_string = ' ˄ '.join(map(lambda cond: str(self._A_star[cond]),self._A_star))
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
        self.prune_condition_map()
        n_uniq_ant = len(self.condition_map)
        n_total_ant = 0
        for rule in self.rules:
            n_total_ant += len(rule.A())
        return len(self.rules), n_uniq_ant, n_total_ant

    def compute_classification_performance(self, X, y_true):
        y_pred = self.predict(X)
        self.geometric_mean_score = geometric_mean_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.accuracy_score = accuracy_score(y_true, y_pred)

    def predict(self, X):
        return np.ravel(self._predict(X)[0])

    def predict_proba(self, X):
        return self._predict(X, proba=True)[0]

    def _predict(self, X, proba=False):
        if proba:
            prediction = np.empty((X.shape[0], self.rules[0].class_dist.shape[0]))#np.zeros((X.shape[0], self.rules[0].class_dist.shape[0]))
        else:
            prediction = np.empty((X.shape[0], self.rules[0].y.shape[0]))#np.zeros((X.shape[0], self.rules[0].y.shape[0]))

        covered_mask = np.zeros((X.shape[0],), dtype=bool)  # records the records that are already covered by some rule
        for i, rule in enumerate(self.rules):
            r_pred, r_mask = rule.predict(X, condition_map=self.condition_map, proba=proba)
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
        if len(self.A) == 0:  # default rule
            # if there is no conditions in the rule, all the records satisfy and
            # an array of predicted class or class distribution for this rule is returned
            if proba:
                return np.tile(self.class_dist, (X.shape[0], self.class_dist.shape[0])), \
                       np.ones((X.shape[0]), dtype=bool)
            else:
                return np.tile(self.y, (X.shape[0], self.y.shape[0])), \
                       np.ones((X.shape[0]), dtype=bool)
        # apply the and operator to all satifies_array functions of the conditions of this rule

        mask = reduce(operator.and_, [condition_map[cond].satisfies_array(X) for cond in self.A])
        if proba:
            prediction = np.zeros((X.shape[0], self.class_dist.shape[0]))
            prediction[mask] = self.class_dist
        else:
            prediction = np.zeros((X.shape[0], self.y.shape[0]))
            prediction[mask] = self.y
        return prediction, mask

    def get_condition(self, att_index):
        for cond in self.A:
            if cond.get_attribute_index() == att_index:
                return cond
        return None

    # def A_(self, frozen=True):
    #     if frozen:
    #         return self.A
    #     else:
    #         return set(self.A)

    # def set_A(self, conditions):
    #     self.A = frozenset(conditions)

    # def y(self):
    #     return self.y
    #
    # def class_index(self):
    #     return self.class_index
    #
    # def class_distribution(self):
    #     return self.class_dist
    #
    # def logit_score(self):
    #     return self.logit_score
    #
    # def n_samples(self):
    #     return self.n_samples
    #
    # def cov(self):
    #     return self.cov
    #
    # def supp(self):
    #     return self.supp
    #
    # def conf(self):
    #     return self.conf
    #
    # def weight(self):
    #     return self.weight

    def add_condition(self, condition):
        if condition not in self.A:
            self.A.add(condition)

    # def set_measures(self, cov, conf, supp):
    #     self.cov = cov
    #     self.conf = conf
    #     self.supp = supp

    def __init__(self, conditions, class_dist=None, logit_score=None, y=None, y_class_index=None, n_samples=0,
                 n_outputs=1, classes=None, weight=0):
        if class_dist is None:
            class_dist = np.array([0.5, 0.5])
        self.A = conditions  # conditions
        self.class_dist = class_dist
        self.logit_score = logit_score
        self.y = y
        self.class_index = y_class_index
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.classes = list(classes)
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
        return self.op(value, self.value)

    def satisfies_array(self, arr):
        # apply the operator to the values in arr of the column equal to the index of the attribute of this condition
        # and returns an array of bool
        return self.op(arr[:, self.att_index], self.value)

    # def attribute_index(self):
    #     return self._att_index
    #
    # def operator(self):
    #     return self._op
    #
    # def value(self):
    #     return self._value

    def __init__(self, att_index, op, value, att_name=None):
        self.att_index = int(att_index)
        if att_name is None:
            self.att_name = 'att_' + str(att_index)
        else:
            self.att_name = att_name
        self.op = op
        self.value = float(value)

    def __str__(self):
        return '({} {} {:.3f})'.format(self.att_name, op_dict[self.op.__name__], self.value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
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

