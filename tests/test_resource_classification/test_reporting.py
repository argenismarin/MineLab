"""Tests for minelab.resource_classification.reporting."""

import numpy as np
import pytest

from minelab.resource_classification.reporting import (
    grade_tonnage_by_category,
    resource_statement,
)


class TestResourceStatement:
    """Tests for resource statement."""

    def test_above_cutoff(self):
        """Only blocks above cutoff should be reported."""
        tonnages = np.array([1000, 1000, 1000, 1000])
        grades = np.array([2.0, 0.3, 1.5, 0.1])
        classification = np.array([1, 1, 2, 3])
        result = resource_statement(tonnages, grades, classification, 0.5)
        # Only blocks with grade >= 0.5: indices 0, 2
        total_tonnes = sum(
            cat["tonnes"] for cat in result.values()
            if isinstance(cat, dict) and "tonnes" in cat
        )
        assert total_tonnes == pytest.approx(2000, rel=0.1)

    def test_categories_present(self):
        """Should have measured, indicated, inferred keys."""
        tonnages = np.array([1000, 1000, 1000])
        grades = np.array([2.0, 1.5, 0.8])
        classification = np.array([1, 2, 3])
        result = resource_statement(tonnages, grades, classification, 0.5)
        assert "measured" in result
        assert "indicated" in result
        assert "inferred" in result


class TestGradeTonnageByCategory:
    """Tests for grade-tonnage curves by category."""

    def test_returns_cutoffs(self):
        """Should return curves at specified cutoffs."""
        tonnages = np.array([1000, 1000, 1000])
        grades = np.array([2.0, 1.5, 0.8])
        classification = np.array([1, 2, 3])
        cutoffs = np.array([0.5, 1.0, 1.5, 2.0])
        result = grade_tonnage_by_category(tonnages, grades, classification, cutoffs)
        assert "cutoffs" in result
        assert len(result["cutoffs"]) == 4

    def test_tonnage_decreases(self):
        """Tonnage should decrease with increasing cutoff."""
        tonnages = np.array([1000] * 10)
        grades = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
        classification = np.array([1] * 10)
        cutoffs = np.array([0.5, 1.0, 2.0])
        result = grade_tonnage_by_category(tonnages, grades, classification, cutoffs)
        measured = result["measured"]
        assert measured["tonnes"][0] >= measured["tonnes"][1]
        assert measured["tonnes"][1] >= measured["tonnes"][2]


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestResourceStatementValidation:
    """Validation tests for resource_statement."""

    def test_empty_arrays_raises(self):
        """Empty input arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            resource_statement(np.array([]), np.array([]), np.array([]), 0.5)

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            resource_statement(
                np.array([1000, 2000]),
                np.array([1.0]),
                np.array([1]),
                0.5,
            )


class TestGradeTonnageByCategoryValidation:
    """Validation tests for grade_tonnage_by_category."""

    def test_empty_arrays_raises(self):
        """Empty input arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            grade_tonnage_by_category(
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([0.5]),
            )

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            grade_tonnage_by_category(
                np.array([1000, 2000]),
                np.array([1.0]),
                np.array([1]),
                np.array([0.5]),
            )

    def test_empty_cutoffs_raises(self):
        """Empty cutoffs array should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            grade_tonnage_by_category(
                np.array([1000]),
                np.array([1.0]),
                np.array([1]),
                np.array([]),
            )
