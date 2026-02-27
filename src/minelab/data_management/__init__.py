"""Drillhole management, compositing, desurvey, and I/O formats."""

from minelab.data_management.compositing import (
    composite_by_bench,
    composite_by_geology,
    composite_by_length,
)
from minelab.data_management.desurvey import (
    balanced_tangential,
    compute_coordinates,
    minimum_curvature,
    tangential,
)
from minelab.data_management.drillholes import DrillholeDB
from minelab.data_management.io_formats import (
    export_block_model_csv,
    read_csv_drillholes,
    read_gslib,
    write_gslib,
)
from minelab.data_management.validation import (
    check_assay_gaps,
    check_assay_overlaps,
    check_collar_duplicates,
    check_survey_consistency,
    validation_report,
)

__all__ = [
    # drillholes
    "DrillholeDB",
    # compositing
    "composite_by_length",
    "composite_by_geology",
    "composite_by_bench",
    # desurvey
    "minimum_curvature",
    "tangential",
    "balanced_tangential",
    "compute_coordinates",
    # io_formats
    "read_gslib",
    "write_gslib",
    "read_csv_drillholes",
    "export_block_model_csv",
    # validation
    "check_collar_duplicates",
    "check_survey_consistency",
    "check_assay_overlaps",
    "check_assay_gaps",
    "validation_report",
]
