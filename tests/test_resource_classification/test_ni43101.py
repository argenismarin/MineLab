"""Tests for minelab.resource_classification.ni43101."""

import numpy as np
import pytest

from minelab.resource_classification.ni43101 import (
    ni43101_classify,
)


class TestNI43101Classify:
    """Tests for NI 43-101 classification."""

    def test_returns_categories(self):
        """Should return array of categories."""
        spacing = np.array([20, 40, 80])
        continuity = np.array([0.9, 0.6, 0.3])
        confidence = np.array([0.95, 0.7, 0.4])
        result = ni43101_classify(spacing, continuity, confidence)
        assert len(result) == 3

    def test_categories_in_range(self):
        """Categories should be 1, 2, or 3."""
        spacing = np.array([20, 40, 80])
        continuity = np.array([0.9, 0.6, 0.3])
        confidence = np.array([0.95, 0.7, 0.4])
        result = ni43101_classify(spacing, continuity, confidence)
        assert np.all((result >= 1) & (result <= 3))


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestNI43101ClassifyValidation:
    """Validation tests for ni43101_classify."""

    def test_empty_arrays_raises(self):
        """Empty input arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            ni43101_classify(np.array([]), np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ni43101_classify(np.array([20, 40]), np.array([0.9]), np.array([0.95]))

    def test_missing_threshold_key_raises(self):
        """Custom thresholds missing 'measured' key should raise ValueError."""
        bad_thresholds = {
            "indicated": {"spacing_max": 50, "continuity_min": 0.5, "confidence_min": 0.7},
        }
        with pytest.raises(ValueError, match="measured"):
            ni43101_classify(
                np.array([20]),
                np.array([0.9]),
                np.array([0.95]),
                thresholds=bad_thresholds,
            )

    def test_missing_subkey_raises(self):
        """Custom thresholds with missing sub-key should raise ValueError."""
        bad_thresholds = {
            "measured": {"spacing_max": 25, "continuity_min": 0.8},
            "indicated": {"spacing_max": 50, "continuity_min": 0.5, "confidence_min": 0.7},
        }
        with pytest.raises(ValueError, match="confidence_min"):
            ni43101_classify(
                np.array([20]),
                np.array([0.9]),
                np.array([0.95]),
                thresholds=bad_thresholds,
            )
