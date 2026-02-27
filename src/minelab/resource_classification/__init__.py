"""JORC 2012, NI 43-101 resource classification and reporting."""

from minelab.resource_classification.classification_criteria import (
    classify_by_kriging_variance,
    classify_by_search_pass,
    slope_of_regression,
)
from minelab.resource_classification.jorc import (
    jorc_classify,
    jorc_table1,
)
from minelab.resource_classification.ni43101 import (
    ni43101_classify,
)
from minelab.resource_classification.reporting import (
    grade_tonnage_by_category,
    resource_statement,
)

__all__ = [
    # jorc
    "jorc_classify",
    "jorc_table1",
    # ni43101
    "ni43101_classify",
    # classification_criteria
    "classify_by_kriging_variance",
    "classify_by_search_pass",
    "slope_of_regression",
    # reporting
    "resource_statement",
    "grade_tonnage_by_category",
]
