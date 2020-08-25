from enum import Enum
import numpy as np


class RuleSet:

    def __init__(self, ruleset):
        self._rules = ruleset.get_rule_list()
        self._condition_map = ruleset.get_condition_map()
        self.n_total_ant = ruleset.n_total_ant
        self.n_uniq_ant = ruleset.n_uniq_ant

    def __init__(self, rules=[], condition_map={}):
        self._rules = rules
        self._condition_map = condition_map
        self.n_total_ant = 0
        self.n_uniq_ant = 0

    def set_condition_map(self, condition_map):
        self._condition_map = condition_map

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
        i = 1;
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
        n_uniq_ant = len(self._condition_map)
        n_total_ant = 0
        for rule in self._rules:
            n_total_ant += len(rule._A)
        return len(self._rules), n_uniq_ant / n_total_ant


class Rule:

    def covers(self, instance):
        for att_index, value in instance.items():
            cond = self.get_condition(int(att_index))
            if cond is None or not cond.satisfies(value):
                return False
        return True

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

    def add_condition(self, condition):
        if condition not in self._A:
            self._A.add(condition)

    def set_measures(self, cov, conf, supp):
        self._cov = cov
        self._conf = conf
        self._supp = supp

    def __init__(self, conditions, class_dist=[0.5, 0.5], logit_score=None, y=None, y_class_index=None, n_samples=0,
                 n_outputs=1, classes=None):
        self._A = conditions  # conditions
        self._class_dist = list(class_dist)
        self._logit_score = logit_score
        self._y = list(y)
        self._class_index = y_class_index
        self._n_samples = n_samples
        self._n_outputs = n_outputs
        self._classes = list(classes)
        self._cov = 0
        self._supp = 0
        self._conf = 0

    def __str__(self):
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


class Operator(Enum):
    EQUAL_TO = '='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    GREATER_OR_EQUAL_THAN = '≥'
    LESS_OR_EQUAL_THAN = '≤'
    DIFFERENT_THAN = '!='


class Condition:

    def satisfies(self, value):
        if self._operator == Operator.EQUAL_TO:
            return value == self._value
        if self._operator == Operator.GREATER_THAN:
            return value > self._value
        if self._operator == Operator.LESS_THAN:
            return value < self._value
        if self._operator == Operator.GREATER_OR_EQUAL_THAN:
            return value >= self._value
        if self._operator == Operator.LESS_OR_EQUAL_THAN:
            return value <= self._value
        if self._operator == Operator.DIFFERENT_THAN:
            return value != self._value

    def attribute_index(self):
        # return str(self._att_index) + str(ord(self._operator)) + str(self._value)
        return self._att_index

    def operator(self):
        return self._operator

    def value(self):
        return self._value

    def __init__(self, att_index, operator, value, att_name=None):
        self._att_index = int(att_index)
        if att_name is None:
            self._att_name = 'att_' + str(att_index)
        else:
            self._att_name = att_name
        self._operator = operator
        self._value = float(value)

    def __str__(self):
        # return '(' + self._att_name+' ' + self._operator.value + ' ' + str(self._value) + ')'
        return '({} {} {:.3f})'.format(self._att_name, self._operator.value, self._value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Condition):
            if hash(self) == hash(other):
                return True
            if self._att_index != other._att_index:
                return False
            if self._operator != other._operator:
                return False
            return self._value == other._value
        return False

    def __hash__(self):
        return hash((self._att_index, self._operator, self._value))
