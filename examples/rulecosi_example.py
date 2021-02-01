from sklearn.datasets import load_iris
from rulecosi import RuleCOSIClassifier
from rulecosi.helpers import total_n_rules
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd

k_folds = 3
repetitions = 1
random_state = 1212
test_fold = 1
np.random.seed(random_state)

random_state_obj = np.random.RandomState(seed=random_state)
random_state_obj_models = np.random.RandomState(seed=random_state)
cv_exps = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state_obj)
data = pd.read_csv('data\pageblocks0.csv')
X = data[data.columns[:-1]].to_numpy()
y = np.ravel(data[data.columns[-1:]])
for k_fold, (train_idx, test_idx) in enumerate(cv_exps.split(X, y)):
    if test_fold == test_fold:  # k_fold:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1212)
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # rule_clf = RuleCOSIClassifier(base_ensemble=CatBoostClassifier(n_estimators=20, verbose=0, random_state=1212), random_state=1212, max_rule_depth=10, rule_order='conf')
        rule_clf = RuleCOSIClassifier(base_ensemble=BaggingClassifier(random_state=random_state),
                                      n_estimators=25, tree_max_depth=6, random_state=random_state_obj_models,
                                      rule_order='cov', conf_threshold=0.75)
        # rule_clf = RuleCOSIClassifier(base_ensemble=XGBClassifier(random_state=random_state), n_estimators=10, tree_max_depth=2, random_state=1212, rule_max_depth=10, rule_order='conf')

        # rule_clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(n_estimators=5), random_state=1212, max_rule_depth=10)
        # rule_clf = RuleCOSIClassifier(base_ensemble=RandomForestClassifier(n_estimators=5,random_state=8504), random_state=1212, max_rule_depth=10)
        # rule_clf = RuleCOSIClassifier(base_ensemble=BaggingClassifier(n_estimators=5,random_state=1212),random_state=1212, max_rule_depth=10)
        # rule_clf = RuleCOSIClassifier(base_ensemble=AdaBoostClassifier(n_estimators=5, random_state=1212),random_state=1212, max_rule_depth=10, rule_order='conf')
        rule_clf.fit(X_train, y_train)

        rule_clf.print_rules(verbose=1)
        y_pred = rule_clf.predict(X_test)
        # y_pred_ens = rule_clf._base_ens.predict(X_test,validate_features=False)
        if isinstance(rule_clf.base_ensemble, XGBClassifier):
            y_pred_ens = rule_clf.base_ensemble.predict(X_test, validate_features=False)
        else:
            y_pred_ens = rule_clf.base_ensemble.predict(X_test)
        print("Combinations: {}".format(rule_clf._n_combinations))
        print("============  Original ensemble  =============")
        print(classification_report(y_test, y_pred_ens))
        print("============  RuleCOSI =======================")
        print(classification_report(y_test, y_pred))
        stop = []
