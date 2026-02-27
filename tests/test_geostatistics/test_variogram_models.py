"""Tests for minelab.geostatistics.variogram_models."""

import numpy as np
import pytest

from minelab.geostatistics.variogram_models import (
    exponential,
    gaussian,
    hole_effect,
    nested_model,
    nugget_effect,
    power,
    spherical,
)


class TestSpherical:
    """Tests for spherical variogram model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert spherical(0, 0, 10, 100) == 0.0

    def test_known_value(self):
        """Isaaks & Srivastava 1989: spherical(50, 0, 10, 100) = 6.875."""
        assert spherical(50, 0, 10, 100) == pytest.approx(6.875, rel=1e-6)

    def test_at_range(self):
        """At h = a, γ should equal the sill."""
        assert spherical(100, 0, 10, 100) == pytest.approx(10.0, rel=1e-6)

    def test_beyond_range(self):
        """Beyond the range, γ should stay at sill."""
        assert spherical(150, 0, 10, 100) == pytest.approx(10.0, rel=1e-6)

    def test_with_nugget(self):
        """With nugget C0=2, sill=10: γ(50) = 2 + 8*(1.5*0.5 - 0.5*0.125) = 7.5."""
        result = spherical(50, 2, 10, 100)
        expected = 2.0 + 8.0 * (1.5 * 0.5 - 0.5 * 0.5**3)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_array_input(self):
        """Vectorized over array of lags."""
        h = np.array([0, 50, 100, 150])
        result = spherical(h, 0, 10, 100)
        expected = np.array([0.0, 6.875, 10.0, 10.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_invalid_sill(self):
        """Negative sill must raise ValueError."""
        with pytest.raises(ValueError):
            spherical(50, 0, -1, 100)

    def test_invalid_range(self):
        """Zero range must raise ValueError."""
        with pytest.raises(ValueError):
            spherical(50, 0, 10, 0)


class TestExponential:
    """Tests for exponential variogram model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert exponential(0, 0, 10, 100) == 0.0

    def test_known_value(self):
        """Cressie 1993: exponential(100, 0, 10, 100) ≈ 9.502."""
        result = exponential(100, 0, 10, 100)
        expected = 10.0 * (1.0 - np.exp(-3.0))
        assert result == pytest.approx(expected, rel=1e-4)
        assert result == pytest.approx(9.502, rel=1e-3)

    def test_approaches_sill(self):
        """At very large h, γ → sill."""
        assert exponential(10000, 0, 10, 100) == pytest.approx(10.0, abs=0.01)

    def test_with_nugget(self):
        """γ(0) = 0 even with nugget; γ(h>0) includes nugget."""
        assert exponential(0, 2, 10, 100) == 0.0
        result = exponential(100, 2, 10, 100)
        assert result > 2.0

    def test_array_input(self):
        """Vectorized input."""
        h = np.array([0, 50, 100])
        result = exponential(h, 0, 10, 100)
        assert result[0] == 0.0
        assert result[1] < result[2]


class TestGaussian:
    """Tests for gaussian variogram model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert gaussian(0, 0, 10, 100) == 0.0

    def test_known_value(self):
        """Cressie 1993: gaussian(100, 0, 10, 100) ≈ 9.502."""
        result = gaussian(100, 0, 10, 100)
        expected = 10.0 * (1.0 - np.exp(-3.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_smooth_at_origin(self):
        """Gaussian is parabolic at origin — very small h gives very small γ."""
        val_small = gaussian(1, 0, 10, 100)
        val_exp = exponential(1, 0, 10, 100)
        # Gaussian should be much smaller than exponential near origin
        assert val_small < val_exp

    def test_approaches_sill(self):
        """At very large h, γ → sill."""
        assert gaussian(10000, 0, 10, 100) == pytest.approx(10.0, abs=0.01)


class TestPower:
    """Tests for power variogram model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert power(0, 0, 1.5, 1.0) == 0.0

    def test_linear(self):
        """With exponent=1, power model is linear: γ(h)=b*h."""
        assert power(10, 0, 1.5, 1.0) == pytest.approx(15.0, rel=1e-6)

    def test_unbounded(self):
        """Power model is unbounded — larger h gives larger γ."""
        v1 = power(100, 0, 1.0, 1.5)
        v2 = power(200, 0, 1.0, 1.5)
        assert v2 > v1

    def test_invalid_exponent_low(self):
        """Exponent must be > 0."""
        with pytest.raises(ValueError, match="exponent"):
            power(10, 0, 1.0, 0.0)

    def test_invalid_exponent_high(self):
        """Exponent must be < 2."""
        with pytest.raises(ValueError, match="exponent"):
            power(10, 0, 1.0, 2.0)


class TestNuggetEffect:
    """Tests for pure nugget effect model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert nugget_effect(0, 5.0) == 0.0

    def test_nonzero_lag(self):
        """γ(h>0) must equal nugget."""
        assert nugget_effect(10, 5.0) == pytest.approx(5.0, rel=1e-6)
        assert nugget_effect(0.001, 5.0) == pytest.approx(5.0, rel=1e-6)

    def test_array_input(self):
        """Vectorized input."""
        h = np.array([0, 1, 10, 100])
        result = nugget_effect(h, 3.0)
        expected = np.array([0.0, 3.0, 3.0, 3.0])
        np.testing.assert_allclose(result, expected)


class TestHoleEffect:
    """Tests for hole-effect variogram model."""

    def test_at_origin(self):
        """γ(0) must be 0."""
        assert hole_effect(0, 0, 10, 100) == 0.0

    def test_oscillation(self):
        """Hole-effect model oscillates — γ can dip below and above sill."""
        lags = np.linspace(1, 500, 500)
        gamma = hole_effect(lags, 0, 10, 100)
        # It should oscillate: check that it both exceeds and falls below sill
        assert np.any(gamma > 10.0)  # overshoots
        assert np.any(gamma < 10.0)  # dips

    def test_approaches_sill(self):
        """At very large h, oscillation dampens toward sill."""
        val = hole_effect(10000, 0, 10, 100)
        assert val == pytest.approx(10.0, abs=0.1)


class TestNestedModel:
    """Tests for nested (composite) variogram model."""

    def test_single_structure(self):
        """Nested with one spherical = spherical alone."""
        structs = [{"model": "spherical", "nugget": 0, "sill": 10, "range_a": 100}]
        result = nested_model(50, structs)
        expected = spherical(50, 0, 10, 100)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_two_structures(self):
        """Sum of spherical + exponential."""
        structs = [
            {"model": "spherical", "nugget": 0, "sill": 5, "range_a": 100},
            {"model": "exponential", "nugget": 0, "sill": 5, "range_a": 100},
        ]
        result = nested_model(50, structs)
        expected = spherical(50, 0, 5, 100) + exponential(50, 0, 5, 100)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_at_origin(self):
        """Nested model at h=0 should be 0."""
        structs = [
            {"model": "spherical", "nugget": 0, "sill": 5, "range_a": 100},
            {"model": "exponential", "nugget": 0, "sill": 5, "range_a": 100},
        ]
        assert nested_model(0, structs) == 0.0

    def test_empty_structures(self):
        """Empty structures list must raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            nested_model(50, [])

    def test_unknown_model(self):
        """Unknown model string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            nested_model(50, [{"model": "invalid", "nugget": 0, "sill": 5}])

    def test_array_input(self):
        """Vectorized input with nested model."""
        structs = [
            {"model": "spherical", "nugget": 0, "sill": 5, "range_a": 100},
            {"model": "exponential", "nugget": 0, "sill": 5, "range_a": 100},
        ]
        h = np.array([0, 50, 100])
        result = nested_model(h, structs)
        assert result[0] == 0.0
        assert len(result) == 3
