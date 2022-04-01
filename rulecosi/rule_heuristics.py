""" Class used for  for measuring the heuristics of the rules

"""

import operator

from functools import reduce
from gmpy2 import popcount
from sys import byteorder
import numpy as np


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
        self.cov_threshold = cov_threshold  # remove
        self.conf_threshold = conf_threshold
        self.min_samples = min_samples  # remove

        self.training_bit_sets = None
        self._cond_cov_dict = None

        self.training_heuristics_dict = None

        self.ones = int.from_bytes(
            np.packbits(np.ones(self.X.shape[0], dtype=bool)),
            byteorder=byteorder)
        self.zeros = int.from_bytes(
            np.packbits(np.zeros(self.X.shape[0], dtype=bool)),
            byteorder=byteorder)

    def get_conditions_heuristics(self, conditions,
                                  uncovered_mask=None):
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
        heuristics_dict = self.create_empty_heuristics_dict()
        if len(conditions) == 0:
            return self.get_training_heuristics_dict(
                uncovered_mask=uncovered_mask), uncovered_mask

        b_array_conds = [reduce(operator.and_,
                                [self._cond_cov_dict[i][cond[0]] for cond in
                                 conditions])
                         for i in range(len(self.classes_))]
        b_array_conds.append(reduce(operator.or_, [i for i in b_array_conds]))

        if uncovered_mask is not None:
            b_array_conds = [b_array_measure & uncovered_mask for
                             b_array_measure in b_array_conds]

            updated_mask = ~b_array_conds[-1] & uncovered_mask
            # uncovered_mask.clear()
            uncovered_mask = updated_mask
        cov_count = popcount(b_array_conds[-1])  # .bit_count() #.sum()
        if cov_count == 0:
            heuristics_dict['cov_set'] = [self.zeros for
                                          _ in range(len(self.classes_) + 1)]
            # int.from_bytes(np.ones(self.X.shape[0]) ,
            #                byteorder='big')
            return heuristics_dict, uncovered_mask

        class_cov_count = [popcount(b_array_conds[i]) for i in
                           range(len(self.classes_))]
        coverage = cov_count / self.X.shape[0]

        heuristics_dict['cov_set'] = b_array_conds
        heuristics_dict['cov'] = coverage
        heuristics_dict['cov_count'] = cov_count
        heuristics_dict['class_cov_count'] = class_cov_count
        heuristics_dict['conf'] = [class_count / cov_count for class_count in
                                   class_cov_count]
        heuristics_dict['supp'] = [class_count / self.X.shape[0] for class_count
                                   in class_cov_count]

        return heuristics_dict, uncovered_mask

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
            # uncovered_mask = int.from_bytes(
            #     np.ones(self.X.shape[0]) * np.uint8(255), byteorder=byteorder)
            uncovered_mask = self.ones

        if sequential_covering:
            accurate_rules = []
            local_uncovered_instances = uncovered_mask
            for rule in ruleset:

                # heuristics_dict, uncovered_mask = self.get_conditions_heuristics(
                #     rule.A,
                #     uncovered_mask=uncovered_mask)
                # rule.set_heuristics(heuristics_dict)
                #
                result, uncovered_instances_with_rule = self.rule_is_accurate(
                    rule,
                    local_uncovered_instances)
                if result:
                    accurate_rules.append(rule)
                    local_uncovered_instances = uncovered_instances_with_rule

            ruleset.rules[:] = accurate_rules

            # ruleset.rules[:] = [rule for rule in ruleset if
            #                      self.rule_is_accurate(rule, uncovered_mask)]
        else:
            for rule in ruleset:
                local_uncovered_instances = uncovered_mask
                heuristics_dict, _ = self.get_conditions_heuristics(
                    rule.A,
                    uncovered_mask=local_uncovered_instances)
                rule.set_heuristics(heuristics_dict)

    def _compute_training_bit_sets(self):
        """ Compute the bitsets of the coverage for the prior class distribution
         of the ensemble according to the training set

        """
        training_bit_set = [
            int.from_bytes(np.packbits((self.y == self.classes_[i])),
                           byteorder=byteorder) for
            i in range(len(self.classes_))]
        # training_bit_set.append(np.bitwise_or.reduce(
        #                                training_bit_set))
        training_bit_set.append(reduce(operator.or_,
                                       training_bit_set))

        return training_bit_set

    def _compute_condition_bit_sets(self):
        """ Compute the bitsets of the coverage of every condition contained in
        the ensemble according to the training set

        """
        # empty sets for each condition coverage class
        cond_cov_dict = [{} for _ in range(len(self.classes_) + 1)]
        for cond_id, cond in self.condition_map.items():
            # compute bitarray for the covered records in X_ by condition cond
            cond_coverage_bitarray = int.from_bytes(
                np.packbits(cond.satisfies_array(self.X)), byteorder=byteorder)
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
        if popcount(uncovered_instances) == 0:
            return False, uncovered_instances
        # local_uncovered_instances = uncovered_instances

        heuristics_dict, local_uncovered_instances = self.get_conditions_heuristics(
            rule.A,
            uncovered_mask=uncovered_instances)
        rule.set_heuristics(heuristics_dict)
        # if rule.cov > self.cov_threshold and rule.conf > self.conf_threshold:
        if rule.conf >= self.conf_threshold:
            # uncovered_instances = local_uncovered_instances
            return True, local_uncovered_instances
        else:
            return False, uncovered_instances

    def create_empty_heuristics_dict(self):
        """ Create an empty dictionary for the heuristics to be computed.

        :return: a dictionary with the heuristics to be computed and populated
        """
        return {'cov_set': None,
                'cov': 0.0,
                'cov_count': 0.0,
                'class_cov_count': [0.0 for _ in self.classes_],
                'conf': [0.0 for _ in self.classes_],
                'supp': [0.0 for _ in self.classes_]}

    def get_training_heuristics_dict(self, uncovered_mask=None):
        """ Create a dictionary with the values of the training heuristics.
        In other words, the heuristics of an empty rule.

        :return: a dictionary with the heuristics to be computed and populated
        """
        if self.training_heuristics_dict is None:
            cov_count = popcount(self.training_bit_sets[-1])
            class_cov_count = [popcount(self.training_bit_sets[i]) for i in
                               range(len(self.classes_))]
            coverage = cov_count / self.X.shape[0]
            train_heur_dict = {'cov_set': self.training_bit_sets[-1],
                               # maybe not -1
                               'cov': coverage,
                               'cov_count': cov_count,
                               'class_cov_count': class_cov_count,
                               'conf': [class_count / cov_count for class_count
                                        in
                                        class_cov_count],
                               'supp': [class_count / self.X.shape[0] for
                                        class_count
                                        in class_cov_count]}
            self.training_heuristics_dict = train_heur_dict
        if uncovered_mask is None:
            return self.training_heuristics_dict
        else:
            if popcount(uncovered_mask) == 0:
                return self.create_empty_heuristics_dict()
            masked_training_heuristics = [b_array_measure & uncovered_mask for
                                          b_array_measure in
                                          self.training_bit_sets]
            cov_count = popcount(masked_training_heuristics[-1])
            class_cov_count = [popcount(masked_training_heuristics[i]) for i in
                               range(len(self.classes_))]
            coverage = cov_count / self.X.shape[0]
            return {'cov_set': masked_training_heuristics,
                    'cov': coverage,
                    'cov_count': cov_count,
                    'class_cov_count': class_cov_count,
                    'conf': [class_count / cov_count for class_count
                             in
                             class_cov_count],
                    'supp': [class_count / self.X.shape[0] for
                             class_count
                             in class_cov_count]}
