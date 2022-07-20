""" Class used for  for measuring the heuristics of the rules

"""

import operator
import numpy as np

from functools import reduce
from .bitarray_backend import PythonIntArray, BitArray


class RuleHeuristics:
    """ This class controls the computation of heuristics of the rules.

    For fast computation we use arrays of bits. At the beginning, an N-size
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
        (beta in the paper)The greater the value the more rules are discarded.
        Default value is 0.0, which it only discards rules with null coverage.

    conf_threshold: float, default=0.5
        Confidence or rule accuracy threshold of a rule to be considered for
        further combinations (alpha in the paper). The greater the value, the
        more rules are discarded. Rules with high confidence are accurate rules.
        Default value is 0.5, which represents rules with higher than random
        guessing accuracy.

    bitarray_backend: string, default='python-int'
        Backend library used for the handling array of bits for heuristics
        computations.



    """

    def __init__(self, X, y, classes_, condition_map,
                 cov_threshold=0.0, conf_threshold=0.5,
                 bitarray_backend='python-int'):
        self.X = X
        self.y = y
        self.classes_ = classes_
        self.n_classes = len(classes_)
        self.condition_map = condition_map
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold

        if bitarray_backend == 'python-int':
            self.bitarray_ = PythonIntArray(X.shape[0], classes_)
        else:
            self.bitarray_ = BitArray(X.shape[0], classes_)

        self.training_bit_sets = None
        self._cond_cov_dict = None

        self.training_heuristics_dict = None

        self.ones = self.bitarray_.generate_ones()
        self.zeros = self.bitarray_.generate_zeros()

    def get_conditions_heuristics(self, conditions,
                                  not_cov_mask=None):
        """ Compute the heuristics of the combination of conditions using the
        bitsets  of each condition from the training set. An intersection
        operation is made and the cardinality of the resultant set is used
        for computing the heuristics

        :param conditions: set of conditions' id

        :param not_cov_mask: if different than None, mask out the records that
         are already covered from the training set. Default is None.

        :return: a dictionary with the following keys and form
                - cov_set : array of bitsets representing the coverage by class
                  and total coverage
                - cov: the coverage of the conditions
                - conf: array of the confidence values of the conditions by
                  class
                - supp: array of the support values of the conditions by class
        """
        empty_list = [0.0] * self.n_classes
        heuristics_dict = {'cov_set': [self.zeros] * (self.n_classes + 1),
                           'cov': 0.0,
                           'cov_count': 0.0,
                           'class_cov_count': empty_list,
                           'conf': empty_list,
                           'supp': empty_list}
        if len(conditions) == 0:
            return self.get_training_heuristics_dict(
                not_cov_mask=not_cov_mask), not_cov_mask

        b_array_conds = [reduce(operator.and_,
                                [self._cond_cov_dict[i][cond[0]] for cond in
                                 conditions])
                         for i in range(self.n_classes)]

        b_array_conds.append(reduce(operator.or_, [i for i in b_array_conds]))

        cov_count = self.bitarray_.get_number_ones(b_array_conds[-1])

        if cov_count == 0:
            return heuristics_dict, not_cov_mask

        class_cov_count = [self.bitarray_.get_number_ones(b_array_conds[i]) for
                           i in
                           range(self.n_classes)]
        coverage = cov_count / self.X.shape[0]
        heuristics_dict['cov_set'] = b_array_conds
        heuristics_dict['cov'] = coverage
        heuristics_dict['cov_count'] = cov_count
        heuristics_dict['class_cov_count'] = class_cov_count
        heuristics_dict['conf'] = [class_count / cov_count for class_count in
                                   class_cov_count]
        heuristics_dict['supp'] = [class_count / self.X.shape[0] for class_count
                                   in class_cov_count]

        return heuristics_dict, not_cov_mask

    def compute_rule_heuristics(self, ruleset, not_cov_mask=None,
                                sequential_covering=False, recompute=False):
        """ Compute rule heuristics, but without the sequential_coverage
        parameter, and without removing the rules that do not meet the
        thresholds

        :param ruleset: RuleSet object representing a ruleset

        :param not_cov_mask: if different than None, mask out the records that
            are already covered from the training set. Default is None.

        :param sequential_covering:If true, the covered examples covered by one
            rule are removed. Additionally, if a rule does not meet the
            threshold is discarded. If false, it just compute the heuristics
            with all the records on the training set for all the rules. Default
            is False

        :param recompute: if true, the heuristics are recomputed and set
            in the rule.

        """
        # with this option, the heuristics are completely recomputed
        # without any mask
        if recompute:
            for rule in ruleset:
                heuristics_dict, _ = self.get_conditions_heuristics(
                    rule.A)
                rule.set_heuristics(heuristics_dict)
            return

        if not_cov_mask is None:
            not_cov_mask = self.ones

        if sequential_covering:
            accurate_rules = []
            local_not_cov_samples = not_cov_mask
            for rule in ruleset:

                result, not_cov_samples_with_rule = self.rule_is_accurate(
                    rule,
                    local_not_cov_samples)
                if result:
                    accurate_rules.append(rule)
                    local_not_cov_samples = not_cov_samples_with_rule

            ruleset.rules[:] = accurate_rules

        else:
            for rule in ruleset:
                self.set_rule_heuristics(rule, not_cov_mask)

    def _compute_training_bit_sets(self):
        """ Compute the bitsets of the coverage for the prior class distribution
         of the ensemble according to the training set

        """
        training_bit_set = [
            self.bitarray_.get_array(self.y == self.classes_[i]) for
            i in range(self.n_classes)]
        training_bit_set.append(reduce(operator.or_,
                                       training_bit_set))

        return training_bit_set

    def _compute_condition_bit_sets(self):
        """ Compute the bitsets of the coverage of every condition contained in
        the ensemble according to the training set

        """
        # empty sets for each condition coverage class
        cond_cov_dict = [{} for _ in range(self.n_classes + 1)]
        for cond_id, cond in self.condition_map.items():
            # compute bitarray for the covered records in X_ by condition cond
            cond_coverage_bitarray = self.bitarray_.get_array(
                cond.satisfies_array(self.X))
            # create the entries in the dictionary
            for i in range(self.n_classes):
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

    def rule_is_accurate(self, rule, not_cov_samples):
        """ Determine if a rule meet the coverage and confidence thresholds

        :param rule: a Rule object

        :param not_cov_samples:  mask out the records that are already
            covered from the training set.

        :return: boolean indicating if the rule satisfy the thresholds
        """
        if self.bitarray_.get_number_ones(not_cov_samples) == 0:
            return False, not_cov_samples

        local_not_cov_samples = self.set_rule_heuristics(rule,
                                                             not_cov_samples)

        if rule.conf > self.conf_threshold and rule.cov > self.cov_threshold:
            return True, local_not_cov_samples
        else:
            return False, not_cov_samples

    def create_empty_heuristics_dict(self):
        """ Create an empty dictionary for the heuristics to be computed.

        :return: a dictionary with the heuristics to be computed and populated
        """
        empty_list = [0.0] * self.n_classes
        return {'cov_set': [self.zeros] * (self.n_classes + 1),
                'cov': 0.0,
                'cov_count': 0.0,
                'class_cov_count': empty_list,
                'conf': empty_list,
                'supp': empty_list}

    def get_training_heuristics_dict(self, not_cov_mask=None):
        """ Create a dictionary with the values of the training heuristics.
        In other words, the heuristics of an empty rule.

        :return: a dictionary with the heuristics to be computed and populated
        """
        if self.training_heuristics_dict is None:
            cov_count = self.bitarray_.get_number_ones(
                self.training_bit_sets[-1])
            class_cov_count = [
                self.bitarray_.get_number_ones(self.training_bit_sets[i]) for i
                in
                range(self.n_classes)]
            coverage = cov_count / self.X.shape[0]
            train_heur_dict = {'cov_set': self.training_bit_sets,
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
        if not_cov_mask is None:
            return self.training_heuristics_dict
        else:
            if self.bitarray_.get_number_ones(not_cov_mask) == 0:
                empty_list = [0.0] * self.n_classes
                return {'cov_set': [self.zeros] * (self.n_classes + 1),
                        'cov': 0.0,
                        'cov_count': 0.0,
                        'class_cov_count': empty_list,
                        'conf': empty_list,
                        'supp': empty_list}
            masked_training_heuristics = [b_array_measure & not_cov_mask for
                                          b_array_measure in
                                          self.training_bit_sets]
            cov_count = self.bitarray_.get_number_ones(
                masked_training_heuristics[-1])
            class_cov_count = [
                self.bitarray_.get_number_ones(masked_training_heuristics[i])
                for i in
                range(self.n_classes)]
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

    def combine_heuristics(self, heuristics1, heuristics2):
        cov_set = [self.zeros] * (self.n_classes + 1)
        for i in range(self.n_classes + 1):
            cov_set[i] = heuristics1['cov_set'][i] & heuristics2['cov_set'][i]
        cov_count = self.bitarray_.get_number_ones(cov_set[-1])

        class_cov_count = [self.bitarray_.get_number_ones(cov_set[i]) for i in
                           range(self.n_classes)]

        coverage = cov_count / self.X.shape[0]
        if coverage == 0:
            empty_list = [0.0] * self.n_classes
            return {'cov_set': cov_set,
                    'cov': 0.0,
                    'cov_count': 0.0,
                    'class_cov_count': class_cov_count,
                    'conf': empty_list,
                    'supp': empty_list}
        return {'cov_set': cov_set,
                'cov': coverage,
                'cov_count': cov_count,
                'class_cov_count': class_cov_count,
                'conf': [class_count / cov_count for class_count
                         in
                         class_cov_count],
                'supp': [class_count / self.X.shape[0] for
                         class_count
                         in class_cov_count]

                }

    def set_rule_heuristics(self, rule, mask):
        """ Set the heuristics of the rule.

        :param mask: mask used for computing the heuristics when some samples
        are already covered
        :param rule: rule which the heuristics are set

        """

        mask_cov_set = [cov_set & mask
                        for cov_set in rule.heuristics_dict['cov_set']]

        cov_count = self.bitarray_.get_number_ones(mask_cov_set[-1])

        if cov_count == 0:
            rule.conf = 0.0
            rule.supp = 0.0
            rule.cov = 0.0
            return self.bitarray_.get_complement(mask_cov_set[-1],
                                                 self.ones) & mask

        else:
            class_cov_count = [self.bitarray_.get_number_ones(mask_cov_set[i])
                               for i in
                               range(self.n_classes)]

            coverage = cov_count / self.X.shape[0]

            rule.conf = class_cov_count[rule.class_index] / cov_count
            rule.supp = class_cov_count[rule.class_index] / self.X.shape[0]
            rule.cov = coverage
            rule.n_samples = np.array(class_cov_count)

            return self.bitarray_.get_complement(mask_cov_set[-1],
                                                 self.ones) & mask

