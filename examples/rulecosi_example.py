from sklearn.datasets import load_iris
from rulecosi import RuleCOSIClassifier
from rulecosi.helpers import total_n_rules
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,  BaggingClassifier
import numpy as np
import pandas as pd

binary = True

if binary:
    data = pd.read_csv('data\wisconsin.csv')
    X = data[data.columns[:-1]]
    y = np.ravel(data[data.columns[-1:]])
else:
    iris = load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target



clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(n_estimators=5),random_state=1212)
#clf = RuleCOSIClassifier(base_ensemble=RandomForestClassifier(n_estimators=5),random_state=1212)
#clf = RuleCOSIClassifier(base_ensemble=BaggingClassifier(n_estimators=5),random_state=1212)
#clf = RuleCOSIClassifier(n_estimators=5, random_state=1212)
clf.fit(X, y)

print(clf._original_rulesets[0])
print(clf._original_rulesets[1])
print(clf._original_rulesets[2])
#print(clf._rulesets_test)

