
from rulecosi.rules import RuleSet, Rule, Condition
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

    for idx, ruleset in enumerate(list_of_rulesets):
        filtered_rules = [x for x in ruleset if not not_exists_add(x, x_set)]
        if len(filtered_rules) > 0:
            new_list_rulesets.append(RuleSet(filtered_rules, ruleset.condition_map))
            if weights is not None:
                new_weights.append(weights[idx])
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


def one_bitarray(size):
    b_array = bitarray(size)
    b_array.setall(True)
    return b_array


def list_and_operation(list_):
    return_set = list_[0]
    for i in range(1, len(list_)):
        return_set = return_set & list_[i]
    return return_set


def list_or_operation(list_):
    return_set = list_[0]
    for i in range(1, len(list_)):
        return_set = return_set | list_[i]
    return return_set


def count_rules_conds(ruleset):
    total_cond = 0
    for rule in ruleset:
        total_cond += len(rule.A)
    return len(ruleset.rules), total_cond


# https://stackoverflow.com/questions/54699105/how-to-count-the-number-of-occurrences-of-a-nested-dictionary-key
def count_keys(dict_, key):
    return (key in dict_) + sum(count_keys(v, key) for v in dict_.values() if isinstance(v, dict))

