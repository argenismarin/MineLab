"""Tests for minelab.equipment.truck_cycle."""

import pytest

from minelab.equipment.truck_cycle import (
    rimpull_speed,
    travel_time,
    truck_cycle_time,
)


class TestTruckCycleTime:
    """Tests for truck cycle time calculation."""

    def test_positive(self):
        """Cycle time should be positive."""
        haul = [{"distance": 1000, "speed": 30}]
        ret = [{"distance": 1000, "speed": 40}]
        result = truck_cycle_time(3.0, haul, 1.5, ret, spot_time=0.5)
        assert result["total_time"] > 0

    def test_components_sum(self):
        """Total = load + haul + dump + return + spot + queue."""
        haul = [{"distance": 1000, "speed": 30}]
        ret = [{"distance": 1000, "speed": 40}]
        result = truck_cycle_time(3.0, haul, 1.5, ret, spot_time=0.5, queue_time=1.0)
        expected = 3.0 + result["haul_time"] + 1.5 + result["return_time"] + 0.5 + 1.0
        assert result["total_time"] == pytest.approx(expected, rel=0.01)

    def test_multiple_segments(self):
        """Multiple haul segments."""
        haul = [{"distance": 500, "speed": 30}, {"distance": 500, "speed": 20}]
        ret = [{"distance": 1000, "speed": 40}]
        result = truck_cycle_time(3.0, haul, 1.5, ret)
        assert result["haul_time"] > 0


class TestRimpullSpeed:
    """Tests for rimpull-based speed."""

    def test_positive(self):
        """Speed should be positive."""
        v = rimpull_speed(300, 5, 3, 200)
        assert v > 0

    def test_uphill_slower(self):
        """Steeper grade → lower speed."""
        v_flat = rimpull_speed(300, 0, 3, 200)
        v_uphill = rimpull_speed(300, 10, 3, 200)
        assert v_uphill < v_flat


class TestTravelTime:
    """Tests for segment travel time."""

    def test_positive(self):
        """Travel time should be positive."""
        t = travel_time(1000, 40, 0, 3)
        assert t > 0

    def test_longer_distance_more_time(self):
        """Longer distance → more time."""
        t_short = travel_time(500, 40, 0, 3)
        t_long = travel_time(2000, 40, 0, 3)
        assert t_long > t_short
