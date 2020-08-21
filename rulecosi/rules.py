from enum import Enum
from scipy.special import expit, logsumexp
import numpy as np


class RuleSet:

    def __init__(self, rules=[], condition_map={}):
        self._rules = rules
        self._condition_map = condition_map
        self.n_total_ant = 0
        self.n_uniq_ant = 0

    def set_condition_map(self, condition_map):
        self._condition_map = condition_map

    def get_condition_map(self):
        return self._condition_map

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

    def add_condition(self, condition):
        if condition not in self._A:
            self._A.add(condition)

    def __init__(self, conditions, type_='classifier', value=None, n_samples=0, n_outputs=1, classes=[1, -1]):
        self._A = conditions  # conditions
        if value is None:
            self._value = np.zeros((n_outputs, len(classes))).tolist()
        else:
            self._value = value.tolist()
        # define predicted class possible types {'classifier', 'gradient_classifier', 'regressor'}
        self.type_ = type_
        if type_ == 'classifier':
            #self._y = np.argmax(value, axis=1).tolist()  # consequent
            self._y = [classes[np.argmax(value, axis=1).item()]]
        elif type_ == 'gradient_classifier':
            if len(classes)==2:
                raw_to_proba = expit(self._value)
                if raw_to_proba > 0.5:
                    self._y= [classes[0]]
                else:
                    self._y = [classes[1]]
            else:
                raw_to_proba = expit(self._value)
                self._y = [classes[np.argmax(raw_to_proba, axis=1)]]#.tolist()  # consequent
        elif type == 'regressor':
            self._y = value
        self.n_samples = n_samples
        self._c = []  # combination vector

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

    def get_attribute_index(self):
        # return str(self._att_index) + str(ord(self._operator)) + str(self._value)
        return self._att_index

    def get_id(self):
        return self._att_index

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
