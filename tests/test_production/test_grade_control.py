"""Tests for minelab.production.grade_control."""

import numpy as np
import pytest

from minelab.production.grade_control import (
    information_effect,
    smu_classification,
)


class TestSMUClassification:
    """Tests for SMU classification."""

    def test_all_ore(self):
        """All grades above cutoff → all ore."""
        grades = np.array([2.0, 3.0, 4.0, 5.0])
        result = smu_classification(grades, 1.0)
        assert result["ore_count"] == 4

    def test_all_waste(self):
        """All grades below cutoff → all waste."""
        grades = np.array([0.1, 0.2, 0.3])
        result = smu_classification(grades, 1.0)
        assert result["waste_count"] == 3

    def test_mixed(self):
        """Mixed grades → some ore, some waste."""
        grades = np.array([0.5, 1.5, 0.3, 2.0])
        result = smu_classification(grades, 1.0)
        assert result["ore_count"] == 2
        assert result["waste_count"] == 2


class TestInformationEffect:
    """Tests for information effect quantification."""

    def test_smoothing_effect(self):
        """Estimated grades are smoother → different tonnages."""
        true = np.array([0.5, 1.5, 0.3, 2.0, 0.8])
        estimated = np.array([0.9, 1.2, 0.7, 1.5, 0.9])
        result = information_effect(true, estimated, 1.0)
        assert "tonnage_change_pct" in result

    def test_perfect_estimation(self):
        """Same grades → zero change."""
        grades = np.array([0.5, 1.5, 0.3, 2.0])
        result = information_effect(grades, grades, 1.0)
        assert result["tonnage_change_pct"] == pytest.approx(0.0, abs=0.1)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestSMUClassificationValidation:
    """Validation tests for smu_classification."""

    def test_empty_grades_raises(self):
        """Empty block_grades should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            smu_classification(np.array([]), 1.0)


class TestInformationEffectValidation:
    """Validation tests for information_effect."""

    def test_empty_true_grades_raises(self):
        """Empty true_grades should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            information_effect(np.array([]), np.array([]), 1.0)

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            information_effect(np.array([1.0, 2.0]), np.array([1.0]), 1.0)
