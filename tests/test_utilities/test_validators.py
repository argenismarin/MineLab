"""Tests for minelab.utilities.validators."""

import numpy as np
import pytest

from minelab.utilities.validators import (
    validate_array,
    validate_non_negative,
    validate_percentage,
    validate_positive,
    validate_probabilities,
    validate_range,
)


class TestValidatePositive:
    def test_positive_value(self):
        validate_positive(1.0, "x")  # should not raise

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0, "x")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-5, "x")

    def test_large_positive(self):
        validate_positive(1e12, "x")  # should not raise

    def test_error_message_includes_name(self):
        with pytest.raises(ValueError, match="'my_param'"):
            validate_positive(-1, "my_param")


class TestValidateNonNegative:
    def test_positive_value(self):
        validate_non_negative(1.0, "x")  # should not raise

    def test_zero_ok(self):
        validate_non_negative(0, "x")  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_non_negative(-0.001, "x")


class TestValidateRange:
    def test_in_range(self):
        validate_range(50, 0, 100, "x")  # should not raise

    def test_at_lower_bound(self):
        validate_range(0, 0, 100, "x")  # should not raise

    def test_at_upper_bound(self):
        validate_range(100, 0, 100, "x")  # should not raise

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 100\]"):
            validate_range(-1, 0, 100, "x")

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 100\]"):
            validate_range(101, 0, 100, "x")


class TestValidatePercentage:
    def test_valid_percentage(self):
        validate_percentage(85.5, "recovery")  # should not raise

    def test_zero(self):
        validate_percentage(0, "recovery")  # should not raise

    def test_hundred(self):
        validate_percentage(100, "recovery")  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_percentage(-1, "recovery")

    def test_above_100_raises(self):
        with pytest.raises(ValueError):
            validate_percentage(100.1, "recovery")


class TestValidateArray:
    def test_list_input(self):
        result = validate_array([1.0, 2.0, 3.0], "data")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_numpy_input(self):
        arr = np.array([10, 20])
        result = validate_array(arr, "data")
        np.testing.assert_array_equal(result, [10, 20])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1 element"):
            validate_array([], "data")

    def test_min_length(self):
        with pytest.raises(ValueError, match="at least 5 element"):
            validate_array([1, 2, 3], "data", min_length=5)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric array"):
            validate_array(["a", "b"], "data")

    def test_2d_input_flattened(self):
        result = validate_array([[1, 2], [3, 4]], "data")
        assert result.shape == (4,)


class TestValidateProbabilities:
    def test_valid_probs(self):
        result = validate_probabilities([0.3, 0.7], "p")
        np.testing.assert_allclose(result, [0.3, 0.7])

    def test_sum_not_one_raises(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_probabilities([0.5, 0.6], "p")

    def test_negative_element_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_probabilities([-0.1, 1.1], "p")

    def test_element_above_one_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_probabilities([1.5, -0.5], "p")

    def test_tolerance(self):
        # Should pass: sum = 0.9999999
        validate_probabilities([0.3, 0.3, 0.3999999], "p")
