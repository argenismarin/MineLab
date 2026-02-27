"""Tests for minelab.ventilation.fan_selection."""

import numpy as np
import pytest

from minelab.ventilation.fan_selection import (
    fan_operating_point,
    fan_power,
    fans_in_series_parallel,
)


class TestFanOperatingPoint:
    """Tests for fan operating point."""

    def test_finds_point(self):
        """Should find intersection."""
        Q = np.array([0, 20, 40, 60, 80, 100])
        P = np.array([3000, 2800, 2400, 1800, 1000, 0])
        result = fan_operating_point(Q, P, 0.5)
        assert result["Q_operating"] > 0
        assert result["P_operating"] > 0

    def test_system_curve_match(self):
        """P_operating ≈ R * Q_operating²."""
        Q = np.array([0, 20, 40, 60, 80, 100])
        P = np.array([3000, 2800, 2400, 1800, 1000, 0])
        R = 0.5
        result = fan_operating_point(Q, P, R)
        expected_p = R * result["Q_operating"] ** 2
        assert result["P_operating"] == pytest.approx(expected_p, rel=0.1)


class TestFanPower:
    """Tests for fan power calculation."""

    def test_known_value(self):
        """Q=50, P=2000, η=0.7 → 142857 W."""
        power = fan_power(50, 2000, 0.7)
        assert power == pytest.approx(142857, rel=0.01)

    def test_positive(self):
        """Power should be positive."""
        power = fan_power(30, 1500, 0.65)
        assert power > 0


class TestFansInSeriesParallel:
    """Tests for combined fan characteristics."""

    def test_series_doubles_pressure(self):
        """Two identical fans in series → ~2x pressure at same Q."""
        fan = {"Q": np.array([0, 50, 100]), "P": np.array([2000, 1000, 0])}
        result = fans_in_series_parallel([fan, fan], "series")
        # At Q=50, each fan gives P=1000, series should give ~2000
        idx = np.argmin(np.abs(result["Q"] - 50))
        assert result["P"][idx] == pytest.approx(2000, rel=0.15)

    def test_parallel_doubles_flow(self):
        """Two identical fans in parallel → ~2x flow at same P."""
        fan = {"Q": np.array([0, 50, 100]), "P": np.array([2000, 1000, 0])}
        result = fans_in_series_parallel([fan, fan], "parallel")
        # At P=1000, each fan gives Q=50, parallel should give ~100
        idx = np.argmin(np.abs(result["P"] - 1000))
        assert result["Q"][idx] == pytest.approx(100, rel=0.15)


class TestFanOperatingPointValidation:
    """Validation tests for fan_operating_point."""

    def test_too_few_points(self):
        """Less than 2 Q points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            fan_operating_point(np.array([10]), np.array([100]), 0.5)

    def test_length_mismatch(self):
        """Mismatched Q and P lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            fan_operating_point(np.array([0, 50]), np.array([3000, 2000, 1000]), 0.5)

    def test_no_intersection(self):
        """No intersection should raise ValueError."""
        # System curve R*Q^2 = 100*Q^2 starts at 0 and grows rapidly
        # Fan curve starts at 10 and drops — system always above fan for Q>0
        Q = np.array([10, 20, 30])
        P = np.array([5, 3, 1])
        with pytest.raises(ValueError, match="No intersection"):
            fan_operating_point(Q, P, 100.0)


class TestFanPowerValidation:
    """Validation tests for fan_power."""

    def test_efficiency_above_one(self):
        """Efficiency > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="<= 1"):
            fan_power(50, 2000, 1.5)


class TestFansValidation:
    """Validation tests for fans_in_series_parallel."""

    def test_empty_curves(self):
        """Empty fan_curves should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            fans_in_series_parallel([], "series")

    def test_invalid_config(self):
        """Invalid configuration should raise ValueError."""
        fan = {"Q": np.array([0, 50, 100]), "P": np.array([2000, 1000, 0])}
        with pytest.raises(ValueError, match="series.*parallel"):
            fans_in_series_parallel([fan], "mixed")

    def test_missing_keys(self):
        """Fan curve without Q/P keys should raise ValueError."""
        with pytest.raises(ValueError, match="Q.*P"):
            fans_in_series_parallel([{"flow": np.array([0, 50])}], "series")
