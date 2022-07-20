import pytest

from sklearn.datasets import load_breast_cancer
from rulecosi import RuleCOSIClassifier


@pytest.fixture
def data():
    return load_breast_cancer(return_X_y=True)


def test_rulecosi_classifier(data):
    X, y = data
    clf = RuleCOSIClassifier()

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
