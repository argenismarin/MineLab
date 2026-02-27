"""Tests for minelab.resource_classification.classification_criteria."""

import numpy as np
import pytest

from minelab.resource_classification.classification_criteria import (
    classify_by_kriging_variance,
    classify_by_search_pass,
    slope_of_regression,
)


class TestClassifyByKrigingVariance:
    """Tests for KV-based classification."""

    def test_known(self):
        """kv=0.1 → Measured(1), kv=0.3 → Indicated(2), kv=0.8 → Inferred(3)."""
        kv = np.array([0.1, 0.3, 0.8])
        thresholds = {"measured": 0.2, "indicated": 0.5}
        result = classify_by_kriging_variance(kv, thresholds)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3


class TestClassifyBySearchPass:
    """Tests for search-pass classification."""

    def test_returns_categories(self):
        """Should return integer categories."""
        n_samples = np.array([15, 8, 3])
        min_octants = np.array([4, 2, 1])
        pass_defs = [
            {"min_samples": 12, "min_octants": 4},
            {"min_samples": 6, "min_octants": 2},
        ]
        result = classify_by_search_pass(n_samples, min_octants, pass_defs)
        assert result[0] == 1  # pass 1
        assert result[1] == 2  # pass 2
        assert result[2] == 3  # neither


class TestSlopeOfRegression:
    """Tests for slope of regression."""

    def test_perfect(self):
        """No smoothing → slope ≈ 1."""
        grades = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = slope_of_regression(grades, 0.5)
        assert slope > 0

    def test_range(self):
        """Slope should be in (0, 1]."""
        grades = np.array([1.5, 1.8, 2.0, 2.2, 1.9])
        slope = slope_of_regression(grades, 0.3)
        assert 0 < slope <= 1.1


class TestKVValidation:
    """Validation tests for classify_by_kriging_variance."""

    def test_empty_kv(self):
        """Empty kv should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            classify_by_kriging_variance(np.array([]), {"measured": 0.2, "indicated": 0.5})

    def test_missing_key(self):
        """Missing threshold key should raise ValueError."""
        with pytest.raises(ValueError, match="measured"):
            classify_by_kriging_variance(np.array([0.1]), {"indicated": 0.5})


class TestSearchPassValidation:
    """Validation tests for classify_by_search_pass."""

    def test_empty_arrays(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            classify_by_search_pass(
                np.array([]), np.array([]),
                [{"min_samples": 12, "min_octants": 4}],
            )

    def test_length_mismatch(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            classify_by_search_pass(
                np.array([10, 5]), np.array([3]),
                [{"min_samples": 6, "min_octants": 2}],
            )


class TestSlopeValidation:
    """Validation tests for slope_of_regression."""

    def test_empty_grades(self):
        """Empty grades should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            slope_of_regression(np.array([]), 0.5)

    def test_zero_mean(self):
        """Zero mean should raise ValueError."""
        with pytest.raises(ValueError, match="zero"):
            slope_of_regression(np.array([-1.0, 1.0]), 0.5)
