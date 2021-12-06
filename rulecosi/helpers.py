
from rulecosi.rules import RuleSet, Rule, Condition
from bitarray import bitarray

# def not_exists_add(element, set_):
#     if element not in set_:
#         set_.add(element)
#         return False
#     else:
#         return True


# def remove_duplicated_rules(list_of_rulesets, weights=None):
#     x_set = set()
#     new_list_rulesets = []
#     if weights is not None:
#         new_weights = []
#     else:
#         new_weights = None
#
#     for idx, ruleset in enumerate(list_of_rulesets):
#         filtered_rules = [x for x in ruleset if not not_exists_add(x, x_set)]
#         if len(filtered_rules) > 0:
#             new_list_rulesets.append(RuleSet(filtered_rules, ruleset.condition_map))
#             if weights is not None:
#                 new_weights.append(weights[idx])
#     return new_list_rulesets, new_weights, x_set
#
#
# def order_trees_by_weight(list_of_rulesets, weights):
#     ordered_tuples = [(y, x) for y, x in sorted(zip(weights, list_of_rulesets), reverse=True)]
#     return map(list, zip(*ordered_tuples))


def total_n_rules(list_of_rulesets):
    """ Returns the total number of rules inside each ruleset on a list of
    :class:`RuleSet`

    :param list_of_rulesets: list of :class:`RuleSet`
    :return: total number of rules
    """
    return sum([len(ruleset.get_rule_list()) for ruleset in list_of_rulesets])


# def zero_bitarray(size):
#     b_array = bitarray(size)
#     b_array.setall(False)
#     return b_array


def one_bitarray(size):
    """ Return a bitarray of 1's of the size given by the parameter

    :param size: size of the returning array
    :return: bitarray of 1's
    """
    b_array = bitarray(size)
    b_array.setall(True)
    return b_array


# def list_and_operation(list_):
#     return_set = list_[0]
#     for i in range(1, len(list_)):
#         return_set = return_set & list_[i]
#     return return_set
#
#
# def list_or_operation(list_):
#     return_set = list_[0]
#     for i in range(1, len(list_)):
#         return_set = return_set | list_[i]
#     return return_set
#
#
# def count_rules_conds(ruleset):
#     total_cond = 0
#     for rule in ruleset:
#         total_cond += len(rule.A)
#     return len(ruleset.rules), total_cond


# https://stackoverflow.com/questions/54699105/how-to-count-the-number-of-occurrences-of-a-nested-dictionary-key
def count_keys(dict_, key):
    """ Return the number of times that key occurs in a dictionary and its
    sub dictionaries

    :param dict_: dictionary to be explored
    :param key: key that should be found
    :return: the number of occurrences of the key in _dict
    """
    return (key in dict_) + sum(count_keys(v, key) for v in dict_.values() if isinstance(v, dict))

# taken from https://stackoverflow.com/questions/67655001/how-to-utilize-every-bit-of-a-bytearray-in-python
class BitArray:
    def __init__(self, size):
        self.size = size
        self.bytearray = bytearray((size + 7) >> 3)

    def clear(self):
        ba = self.bytearray
        for i in range(len(ba)):
            ba[i] = 0

    def get_bit(self, bit_ix):
        if not isinstance(bit_ix, int):
            raise IndexError("bit array index not an int")

        if bit_ix < 0 or bit_ix >= self.size:
            raise IndexError("bit array index out of range")

        byte_ix = bit_ix >> 3
        bit_ix = bit_ix & 7
        return (self.bytearray[byte_ix] >> bit_ix) & 1

    def set_bit(self, bit_ix, val):
        if not isinstance(bit_ix, int):
            raise IndexError("bit array index not an int")

        if bit_ix < 0 or bit_ix >= self.size:
            raise IndexError("bit array index out of range")

        if not isinstance(val, int):
            raise ValueError("bit array value not an int")

        if val not in (0, 1):
            raise ValueError("bit array value must be 0 or 1")

        byte_ix = bit_ix >> 3
        bit_ix = bit_ix & 7
        bit_val = 1 << bit_ix

        if val:
            self.bytearray[byte_ix] |= bit_val
        else:
            self.bytearray[byte_ix] &= ~bit_val

    def __getitem__(self, key):
        return self.get_bit(key)

    def __setitem__(self, key, value):
        self.set_bit(key, value)