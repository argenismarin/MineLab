"""Tests for minelab.equipment.productivity."""

import pytest

from minelab.equipment.productivity import (
    excavator_productivity,
    fleet_productivity,
    oee,
)


class TestFleetProductivity:
    """Tests for fleet productivity."""

    def test_positive(self):
        """Productivity should be positive."""
        p = fleet_productivity(5, 100, 30)
        assert p > 0

    def test_more_trucks_higher(self):
        """More trucks → higher productivity."""
        p_small = fleet_productivity(3, 100, 30)
        p_large = fleet_productivity(6, 100, 30)
        assert p_large > p_small


class TestExcavatorProductivity:
    """Tests for excavator productivity."""

    def test_positive(self):
        """Productivity should be positive."""
        p = excavator_productivity(15, 0.85, 25, 1.8)
        assert p > 0

    def test_larger_bucket_higher(self):
        """Larger bucket → higher productivity."""
        p_small = excavator_productivity(10, 0.85, 25, 1.8)
        p_large = excavator_productivity(20, 0.85, 25, 1.8)
        assert p_large > p_small


class TestOEE:
    """Tests for Overall Equipment Effectiveness."""

    def test_known_value(self):
        """A=0.9, U=0.85, E=0.95 → OEE = 0.726."""
        result = oee(0.9, 0.85, 0.95)
        assert result == pytest.approx(0.72675, rel=0.01)

    def test_perfect(self):
        """All 1.0 → OEE = 1.0."""
        result = oee(1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_range(self):
        """OEE should be in [0, 1]."""
        result = oee(0.9, 0.85, 0.95)
        assert 0 < result <= 1
