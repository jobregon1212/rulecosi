from rulecosi.rules import RuleSet, Rule, Condition
#from bitarray import bitarray


def total_n_rules(list_of_rulesets):
    """ Returns the total number of rules inside each ruleset on a list of
    :class:`RuleSet`

    :param list_of_rulesets: list of :class:`RuleSet`
    :return: total number of rules
    """
    return sum([len(ruleset.get_rule_list()) for ruleset in list_of_rulesets])


# def one_bitarray(size):
#     """ Return a bitarray of 1's of the size given by the parameter
#
#     :param size: size of the returning array
#     :return: bitarray of 1's
#     """
#     b_array = bitarray(size)
#     b_array.setall(True)
#     return b_array


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
    return (key in dict_) + sum(
        count_keys(v, key) for v in dict_.values() if isinstance(v, dict))



