"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from rulecosi.rules import Condition, Rule, Operator



class RuleCOSI(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    # """
    # def __init__(self, demo_param='demo_param'):
    #     self.demo_param = demo_param
    #
    # def fit(self, X, y):
    #     """A reference implementation of a fitting function.
    #
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.
    #     y : array-like, shape (n_samples,) or (n_samples, n_outputs)
    #         The target values (class labels in classification, real numbers in
    #         regression).
    #
    #     Returns
    #     -------
    #     self : object
    #         Returns self.
    #     """
    #     X, y = check_X_y(X, y, accept_sparse=True)
    #     self.is_fitted_ = True
    #     # `fit` should always return `self`
    #     return self
    #
    # def predict(self, X):
    #     """ A reference implementation of a predicting function.
    #
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.
    #
    #     Returns
    #     -------
    #     y : ndarray, shape (n_samples,)
    #         Returns an array of ones.
    #     """
    #     X = check_array(X, accept_sparse=True)
    #     check_is_fitted(self, 'is_fitted_')
    #     return np.ones(X.shape[0], dtype=np.int64)


class RuleCOSIClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    #_clf = AdaBoostClassifier()

    def __init__(self,
                 base_estimator=None,
                 base_ensemble=None,
                 n_estimators=5,
                 random_state=None):
        self.base_ensemble = base_ensemble

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        #dt = tree.DecisionTreeClassifier(max_depth=3)
        #_clf = AdaBoostClassifier(n_estimators=20, base_estimator=dt)
        if self.base_ensemble is None:
            self._base_ens = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                                  algorithm="SAMME",
                                  n_estimators=5)
        else:
            self._base_ens = clone(self.base_ensemble)

        self._base_ens.fit(X, y)

        for base_estimator in self._base_ens.estimators_:
            continue

        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self._base_ens.predict(X)

    def get_base_ruleset(self, index=0, print=False):
        check_is_fitted(self, ['X_', 'y_'])

        if index >= len(self._base_ens.estimators_):
            raise ValueError("%s is out of bounds of the base estimators indexes." % index)

        base_dt = self._base_ens.estimators_[index]
        
        #for tree in self._base_ens.estimators_:




    def _traverse_node(self, tree, node_id):
        #first it is a leaf node
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # LESS OR EQUAL THAN (sklearn dt)
            condition = Condition(tree.feature[node_id], Operator.LESS_OR_EQUAL_THAN, tree.threshold[node_id])
            #return [condition.get_id(),]
        pass





