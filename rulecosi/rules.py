from enum import Enum


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
            self._A.append(condition)

    def __init__(self, conditions, consequent):
        self._A = conditions  # conditions
        self._y = consequent  # consequent
        self._c = []  # combination vector

    def __str__(self):
        return_string = ' ˄ '.join(map(str,self._A))
        return_string += ' → ' + self._y
        return return_string

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Rule):
            if id(self) == id(other):
                return True
            if self._A != other._A:
                return False
            return self._y == other._y
        return False


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
        return str(self._att_index) + str(ord(self._operator)) + str(self._value)

    def get_id(self):
        return self._att_index

    def __init__(self, att_index, operator, value, att_name=None):
        self._att_index = att_index
        if att_name is None:
            self._att_name = 'att_' + str(att_index)
        else:
            self._att_name = att_name
        self._operator = operator
        self._value = value

    def __str__(self):
        return '(' + self._att_name+' ' + self._operator.value + ' ' + str(self._value) + ')'

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Condition):
            if id(self) == id(other):
                return True
            if self._att_index != other._att_index:
                return False
            if self._operator != other._operator:
                return False
            return self._value == other._value
        return False


