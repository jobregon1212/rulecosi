import pytest

from sklearn.utils.estimator_checks import check_estimator

from rulecosi import TemplateEstimator
from rulecosi import TemplateClassifier
from rulecosi import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
