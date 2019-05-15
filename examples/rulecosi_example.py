from sklearn.datasets import load_iris
from rulecosi import RuleCOSIClassifier
import numpy as np
import pandas as pd

# iris = load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target

data =pd.read_csv('examples\data\ionosphere.csv')
X = data[data.columns[:-1]]
y = np.ravel(data[data.columns[-1:]])
clf = RuleCOSIClassifier()
clf.fit(X, y)

print(clf)

