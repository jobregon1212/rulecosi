from sklearn.datasets import load_iris
from rulecosi import RuleCOSIClassifier
from rulecosi.helpers import total_n_rules
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,  BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

binary = True

if binary:
    data = pd.read_csv('data\wisconsin.csv')
    #data = pd.read_csv('data\page-blocks0.csv')
    X = data[data.columns[:-1]]
    y = np.ravel(data[data.columns[-1:]])
else:
    iris = load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1212)

rule_clf = RuleCOSIClassifier(base_ensemble=CatBoostClassifier(n_estimators=20, verbose=0, random_state=1212), random_state=1212, max_rule_depth=10, rule_order='conf')
#rule_clf = RuleCOSIClassifier(base_ensemble=LGBMClassifier(n_estimators=5), random_state=1212, max_rule_depth=10, rule_order='conf')
#rule_clf = RuleCOSIClassifier(base_ensemble=XGBClassifier(n_estimators=5), random_state=1212, max_rule_depth=10, rule_order='conf')
#rule_clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(n_estimators=5), random_state=1212, max_rule_depth=10)
#rule_clf = RuleCOSIClassifier(base_ensemble=RandomForestClassifier(n_estimators=5,random_state=8504), random_state=1212, max_rule_depth=10)
#rule_clf = RuleCOSIClassifier(base_ensemble=BaggingClassifier(n_estimators=5,random_state=1212),random_state=1212, max_rule_depth=10)
#rule_clf = RuleCOSIClassifier(base_ensemble=AdaBoostClassifier(n_estimators=5, random_state=1212),random_state=1212, max_rule_depth=10, rule_order='conf')
rule_clf.fit(X_train, y_train)

rule_clf.print_rules(verbose=1)
y_pred = rule_clf.predict(X_test)
#y_pred_ens = rule_clf._base_ens.predict(X_test,validate_features=False)
y_pred_ens = rule_clf.base_ensemble.predict(X_test)
print("Combinations: {}".format(rule_clf._n_combinations))
print("============  Original ensemble  =============")
print(classification_report(y_test, y_pred_ens))
print("============  RuleCOSI =======================")
print(classification_report(y_test, y_pred))
stop =[]


