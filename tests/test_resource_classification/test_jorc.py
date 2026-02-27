"""Tests for minelab.resource_classification.jorc."""

import numpy as np
import pytest

from minelab.resource_classification.jorc import (
    jorc_classify,
    jorc_table1,
)


class TestJORCClassify:
    """Tests for JORC classification."""

    def test_measured(self):
        """Low variance, close spacing → Measured."""
        kv = np.array([0.1, 0.1])
        spacing = np.array([20, 20])
        slope = np.array([0.9, 0.9])
        thresholds = {
            "measured": {"kv_max": 0.2, "spacing_max": 25, "slope_min": 0.8},
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        result = jorc_classify(kv, spacing, slope, thresholds)
        assert np.all(result == 1)  # 1 = Measured

    def test_inferred(self):
        """High variance, wide spacing → Inferred."""
        kv = np.array([0.8])
        spacing = np.array([100])
        slope = np.array([0.3])
        thresholds = {
            "measured": {"kv_max": 0.2, "spacing_max": 25, "slope_min": 0.8},
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        result = jorc_classify(kv, spacing, slope, thresholds)
        assert result[0] == 3  # 3 = Inferred


class TestJORCTable1:
    """Tests for JORC Table 1."""

    def test_summary(self):
        """Should return summary by category."""
        classification = np.array([1, 1, 2, 2, 3])
        tonnages = np.array([1000, 1000, 1000, 1000, 1000])
        grades = np.array([2.0, 2.5, 1.5, 1.8, 0.8])
        result = jorc_table1(classification, tonnages, grades)
        assert "measured" in result
        assert "indicated" in result
        assert "inferred" in result
        assert result["measured"]["tonnes"] > 0


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestJORCClassifyValidation:
    """Validation tests for jorc_classify."""

    def test_empty_arrays_raises(self):
        """Empty input arrays should raise ValueError."""
        thresholds = {
            "measured": {"kv_max": 0.2, "spacing_max": 25, "slope_min": 0.8},
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        with pytest.raises(ValueError, match="empty"):
            jorc_classify(np.array([]), np.array([]), np.array([]), thresholds)

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths should raise ValueError."""
        thresholds = {
            "measured": {"kv_max": 0.2, "spacing_max": 25, "slope_min": 0.8},
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        with pytest.raises(ValueError, match="same length"):
            jorc_classify(np.array([0.1, 0.2]), np.array([20]), np.array([0.9]), thresholds)

    def test_missing_measured_key_raises(self):
        """Thresholds missing 'measured' key should raise ValueError."""
        thresholds = {
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        with pytest.raises(ValueError, match="measured"):
            jorc_classify(np.array([0.1]), np.array([20]), np.array([0.9]), thresholds)

    def test_missing_subkey_raises(self):
        """Thresholds with missing sub-key should raise ValueError."""
        thresholds = {
            "measured": {"kv_max": 0.2, "spacing_max": 25},
            "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
        }
        with pytest.raises(ValueError, match="slope_min"):
            jorc_classify(np.array([0.1]), np.array([20]), np.array([0.9]), thresholds)


class TestJORCTable1Validation:
    """Validation tests for jorc_table1."""

    def test_empty_arrays_raises(self):
        """Empty input arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            jorc_table1(np.array([]), np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            jorc_table1(np.array([1, 2]), np.array([1.0]), np.array([1]))
