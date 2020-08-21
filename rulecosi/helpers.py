from rulecosi.rules import RuleSet
from bitarray import bitarray

def not_exists_add(element, set_):
    if element not in set_:
        set_.add(element)
        return False
    else:
        return True


def remove_duplicated_rules(list_of_rulesets, weights=None):
    x_set = set()
    new_list_rulesets = []
    if weights is not None:
        new_weights = []
    else:
        new_weights = None
    i = 0
    for ruleset in list_of_rulesets:
        filtered_rules = [x for x in ruleset if not not_exists_add(x, x_set)]
        if len(filtered_rules) > 0:
            new_list_rulesets.append(RuleSet(filtered_rules, ruleset.get_condition_map()))
            if weights is not None:
                new_weights.append(weights[i])
        i += 1
    return new_list_rulesets, new_weights, x_set


def order_trees_by_weight(list_of_rulesets, weights):
    ordered_tuples = [(y, x) for y, x in sorted(zip(weights, list_of_rulesets), reverse=True)]
    return map(list, zip(*ordered_tuples))


def total_n_rules(list_of_rulesets):
    return sum([len(ruleset.get_rule_list()) for ruleset in list_of_rulesets])


def zero_bitarray(size):
    b_array = bitarray(size)
    b_array.setall(False)
    return b_array
