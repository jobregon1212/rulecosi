""" Class used for  for measuring the heuristics of the rules

"""

import operator
import copy
from functools import reduce

from bitarray import bitarray

from .helpers import one_bitarray


class RuleHeuristics:
    """ This class controls the computation of heuristics of the rules.

    For fast computation we use the bitarray class. At the beginning, an N-size
    bitarray for each condition is computed, with N=n_samples. This array
    contains 1 if the record was satisfied by the condition and 0 otherwise.
    When a combination is performed, this bitarray are combined using the
    intersection set operation to find out how many  records are covered by the
    new rule (which is a combination of conditions). Additionally, there are two
    extra bitarrays, one covering each of the classes (right now it jus support
    binary class). The cardinality of all these bitarrays are used to compute
    the coverage and confidence of the rules very fast.

     Parameters
    ----------

     X : array-like, shape (n_samples, n_features)
        The training input samples.

    y : array-like, shape (n_samples,)
        The target values. An array of int.

    condition_map: dictionary of <condition_id, Condition>, default=None
        Dictionary of Conditions extracted from all the ensembles. condition_id
        is an integer uniquely identifying the Condition.

    classes : ndarray, shape (n_classes,)
        The classes seen in the ensemble fit method.

    cov_threshold: float, default=0.0
        Coverage threshold of a rule to be considered for further combinations.
        The greater the value the more rules are discarded. Default value is
        0.0, which it only discards rules with null coverage.

    conf_threshold: float, default=0.5
        Confidence or rule accuracy threshold of a rule to be considered for
        further combinations. The greater the value, the more rules are
        discarded. Rules with high confidence are accurate rules. Default value
        is 0.5, which represents rules with higher than random guessing
        accuracy.

    min_samples: int, default=1
        The minimum number of samples required to be at rule in the simplified
        ruleset.

    """

    def __init__(self, X, y, classes_, condition_map,
                 cov_threshold=0.0, conf_threshold=0.5, min_samples=1):
        self.X = X
        self.y = y
        self.classes_ = classes_
        self.condition_map = condition_map
        self.cov_threshold = cov_threshold # remove
        self.conf_threshold = conf_threshold
        self.min_samples = min_samples # remove

        self.training_bit_sets = None
        self._cond_cov_dict = None

    def get_conditions_heuristics(self, conditions, uncovered_mask=None):
        """ Compute the heuristics of the combination of conditions using the
        bitsets  of each condition from the training set. An intersection
        operation is made and the cardinality of the resultant set is used
        for computing the heuristics

        :param conditions: set of conditions' id

        :param uncovered_mask: if different than None, mask out the records that
         are already covered from the training set. Default is None.

        :return: a dictionary with the following keys and form
                - cov_set : array of bitsets representing the coverage by class
                  and total coverage
                - cov: the coverage of the conditions
                - conf: array of the confidence values of the conditions by
                  class
                - supp: array of the support values of the conditions by class
        """
        heuristics_dict = self.create_empty_heuristics()
        if len(conditions) == 0:
            return heuristics_dict
        b_array_conds = [reduce(operator.and_,
                                [self._cond_cov_dict[i][cond[0]] for cond in
                                 conditions])
                         for i in range(len(self.classes_))]
        b_array_conds.append(reduce(operator.or_, [i for i in b_array_conds]))

        if uncovered_mask is not None:
            b_array_conds = [b_array_measure & uncovered_mask for
                             b_array_measure in b_array_conds]

            updated_mask = ~b_array_conds[-1] & uncovered_mask
            uncovered_mask.clear()
            uncovered_mask.extend(updated_mask)
        cov_count = b_array_conds[-1].count()
        if cov_count == 0:
            return heuristics_dict

        class_cov_count = [b_array_conds[i].count() for i in
                           range(len(self.classes_))]
        coverage = cov_count / self.X.shape[0]

        heuristics_dict['cov_set'] = b_array_conds
        heuristics_dict['cov'] = coverage
        heuristics_dict['conf'] = [class_count / cov_count for class_count in
                                   class_cov_count]
        heuristics_dict['supp'] = [class_count / self.X.shape[0] for class_count
                                   in class_cov_count]

        return heuristics_dict

    def compute_rule_heuristics(self, ruleset, uncovered_mask=None,
                                sequential_covering=False):
        """ Compute rule heuristics, but without the sequential_coverage
        parameter, and without removing the rules that do not meet the
        thresholds

        :param ruleset: RuleSet object representing a ruleset

        :param uncovered_mask: if different than None, mask out the records that
            are already covered from the training set. Default is None.

        :param sequential_covering:If true, the covered examples covered by one
            rule are removed. Additionally, if a rule does not meet the
            threshold is discarded. If false, it just compute the heuristics
            with all the records on the training set for all the rules. Default
            is False

        """
        if uncovered_mask is None:
            uncovered_mask = one_bitarray(self.X.shape[0])
        if sequential_covering:
            for rule in ruleset:
               #local_uncovered_instances = copy.copy(uncovered_mask)
               heuristics_dict = self.get_conditions_heuristics(rule.A,
                                                                uncovered_mask=uncovered_mask)
               rule.set_heuristics(heuristics_dict)
            #     self.rule_is_accurate(rule, uncovered_mask)
            # # ruleset.rules[:] = [rule for rule in ruleset if
            # #                     self.rule_is_accurate(rule, uncovered_mask)]
        else:
            for rule in ruleset:
                local_uncovered_instances = copy.copy(uncovered_mask)
                heuristics_dict = self.get_conditions_heuristics(rule.A,
                                                                 uncovered_mask=local_uncovered_instances)
                rule.set_heuristics(heuristics_dict)

    def _compute_training_bit_sets(self):
        """ Compute the bitsets of the coverage for the prior class distribution
         of the ensemble according to the training set

        """
        return [bitarray((self.y == self.classes_[i]).astype(int).tolist()) for
                i in range(len(self.classes_))]

    def _compute_condition_bit_sets(self):
        """ Compute the bitsets of the coverage of every condition contained in
        the ensemble according to the training set

        """
        # empty sets for each condition coverage class
        cond_cov_dict = [{} for _ in range(len(self.classes_) + 1)]
        for cond_id, cond in self.condition_map.items():
            # compute bitarray for the covered records in X_ by condition cond
            cond_coverage_bitarray = bitarray(
                cond.satisfies_array(self.X).astype(int).tolist())
            # create the entries in the dictionary
            for i in range(len(self.classes_)):
                cond_cov_dict[i][cond_id] = cond_coverage_bitarray & \
                                            self.training_bit_sets[i]
            cond_cov_dict[-1][cond_id] = cond_coverage_bitarray
        return cond_cov_dict

    def initialize_sets(self):
        """ Initialize the sets that are going to be used during the combination
        and simplification process This includes the bitsets for the training
        data as well as the bitsets for each of the conditions
        """
        self.training_bit_sets = self._compute_training_bit_sets()
        self._cond_cov_dict = self._compute_condition_bit_sets()

    def rule_is_accurate(self, rule, uncovered_instances):
        """ Determine if a rule meet the coverage and confidence thresholds

        :param rule: a Rule object

        :param uncovered_instances:  mask out the records that are already
            covered from the training set.

        :return: boolean indicating if the rule satisfy the thresholds
        """
        if uncovered_instances.count() == 0:
            return False
        local_uncovered_instances = copy.copy(uncovered_instances)

        heuristics_dict = self.get_conditions_heuristics(rule.A,
                                                         uncovered_mask=local_uncovered_instances)
        rule.set_heuristics(heuristics_dict)
        #if rule.cov > self.cov_threshold and rule.conf > self.conf_threshold:
        if rule.conf > self.conf_threshold:
            uncovered_instances.clear()
            uncovered_instances.extend(local_uncovered_instances)
            return True
        else:
            return False

    def create_empty_heuristics(self):
        """ Create an empty dictionary for the heuristics to be computed.

        :return: a dictionary with the heuristics to be computed and populated
        """
        return {'cov_set': None,
                'cov': 0.0,
                'conf': [0.0 for _ in self.classes_],
                'supp': [0.0 for _ in self.classes_]}
