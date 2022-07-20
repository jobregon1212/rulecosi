import pytest

from sklearn.utils.estimator_checks import check_estimator

from rulecosi import RuleCOSIClassifier


@pytest.mark.parametrize(
    "estimator", [RuleCOSIClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
