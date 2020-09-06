import pytest

from sklearn.utils.estimator_checks import check_estimator

from rulecosi import RuleCOSIClassifier


@pytest.mark.parametrize(
    "Estimator", [RuleCOSIClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
