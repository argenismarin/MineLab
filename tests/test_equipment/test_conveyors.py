"""Tests for minelab.equipment.conveyors."""

import math

import pytest

from minelab.equipment.conveyors import (
    belt_conveyor_capacity,
    belt_tension,
    conveyor_power,
    conveyor_slope_limit,
    idler_spacing,
    screw_conveyor_capacity,
)


class TestBeltConveyorCapacity:
    """Tests for belt_conveyor_capacity."""

    def test_known_value(self):
        """CEMA capacity for 1.2 m belt at 3.5 m/s, 1.8 t/m3, 20 deg."""
        surcharge_rad = math.radians(20)
        area = 0.1 * (1.2 - 0.1) ** 2 * math.tan(surcharge_rad)
        expected = area * 3.5 * 1.8 * 3600
        result = belt_conveyor_capacity(1.2, 3.5, 1.8, 20)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_wider_belt_higher_capacity(self):
        """Wider belt should give higher capacity."""
        q_narrow = belt_conveyor_capacity(0.8, 3.0, 1.6, 20)
        q_wide = belt_conveyor_capacity(1.4, 3.0, 1.6, 20)
        assert q_wide > q_narrow

    def test_faster_speed_higher_capacity(self):
        """Higher belt speed should increase capacity."""
        q_slow = belt_conveyor_capacity(1.0, 2.0, 1.8, 20)
        q_fast = belt_conveyor_capacity(1.0, 4.0, 1.8, 20)
        assert q_fast > q_slow

    def test_invalid_belt_width(self):
        """Negative belt width should raise ValueError."""
        with pytest.raises(ValueError, match="belt_width"):
            belt_conveyor_capacity(-1.0, 3.0, 1.8, 20)

    def test_invalid_surcharge_angle(self):
        """Surcharge angle of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="surcharge_angle"):
            belt_conveyor_capacity(1.0, 3.0, 1.8, 0.0)


class TestConveyorPower:
    """Tests for conveyor_power."""

    def test_known_value(self):
        """Verify power for 500 m conveyor, 30 m lift, 1000 t/h."""
        g = 9.81
        p_h = 0.03 * 500 * 1000 * g / (3600 * 1000)
        p_l = 1000 / 3.6 * g * 30 / 1000
        result = conveyor_power(500, 30, 1000, 0.03)
        assert result["horizontal_power_kw"] == pytest.approx(p_h, rel=1e-4)
        assert result["lift_power_kw"] == pytest.approx(p_l, rel=1e-4)
        assert result["total_power_kw"] == pytest.approx(p_h + p_l, rel=1e-4)

    def test_zero_lift(self):
        """Zero lift should give zero lift power."""
        result = conveyor_power(300, 0.0, 500, 0.03)
        assert result["lift_power_kw"] == pytest.approx(0.0, abs=1e-6)
        assert result["total_power_kw"] == pytest.approx(
            result["horizontal_power_kw"], rel=1e-6
        )

    def test_negative_lift_reduces_power(self):
        """Downhill (negative lift) reduces total power."""
        p_up = conveyor_power(500, 30, 1000, 0.03)
        p_down = conveyor_power(500, -10, 1000, 0.03)
        assert p_down["total_power_kw"] < p_up["total_power_kw"]

    def test_invalid_length(self):
        """Non-positive length should raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            conveyor_power(0, 10, 500, 0.03)

    def test_invalid_capacity(self):
        """Non-positive capacity should raise ValueError."""
        with pytest.raises(ValueError, match="capacity_tph"):
            conveyor_power(300, 10, -100, 0.03)


class TestBeltTension:
    """Tests for belt_tension."""

    def test_known_value(self):
        """Verify belt tension calculation."""
        result = belt_tension(800, 3.0, 0.03, 400, 20)
        assert result["effective_tension_kn"] > 0
        # tight = 1.5 * effective
        assert result["tight_side_kn"] == pytest.approx(
            1.5 * result["effective_tension_kn"], rel=1e-6
        )
        # slack = tight - effective = 0.5 * effective
        assert result["slack_side_kn"] == pytest.approx(
            0.5 * result["effective_tension_kn"], rel=1e-6
        )

    def test_higher_capacity_higher_tension(self):
        """Higher throughput should increase belt tension."""
        t_low = belt_tension(500, 3.0, 0.03, 400, 20)
        t_high = belt_tension(1500, 3.0, 0.03, 400, 20)
        assert (
            t_high["effective_tension_kn"]
            > t_low["effective_tension_kn"]
        )

    def test_invalid_speed(self):
        """Zero speed should raise ValueError."""
        with pytest.raises(ValueError, match="speed"):
            belt_tension(800, 0.0, 0.03, 400, 20)


class TestIdlerSpacing:
    """Tests for idler_spacing."""

    def test_within_range(self):
        """Spacing should be within typical 0.6--1.8 m range."""
        s = idler_spacing(1.2, 1.8, 15.0, 0.02)
        assert 0.6 <= s <= 1.8

    def test_known_value(self):
        """Verify specific computed spacing."""
        raw = min(1.5, 0.02 * 1.2 * 10.0 / (1.8 * 15.0))
        expected = max(0.6, min(1.8, raw))
        result = idler_spacing(1.2, 1.8, 15.0, 0.02)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_heavy_material_shorter_spacing(self):
        """Heavy material should trend toward shorter spacing."""
        s_light = idler_spacing(1.0, 1.0, 10.0, 0.02)
        s_heavy = idler_spacing(1.0, 3.0, 10.0, 0.02)
        assert s_heavy <= s_light

    def test_invalid_density(self):
        """Non-positive density should raise ValueError."""
        with pytest.raises(ValueError, match="material_density"):
            idler_spacing(1.0, -1.0, 10.0, 0.02)


class TestConveyorSlopeLimit:
    """Tests for conveyor_slope_limit."""

    def test_known_value(self):
        """35 deg friction - 5 deg margin = 30 deg."""
        assert conveyor_slope_limit(35.0, 5.0) == 30.0

    def test_zero_margin(self):
        """Zero safety margin returns the friction angle itself."""
        assert conveyor_slope_limit(25.0, 0.0) == 25.0

    def test_margin_exceeds_friction(self):
        """Margin >= friction angle should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            conveyor_slope_limit(10.0, 15.0)

    def test_invalid_friction_angle(self):
        """Non-positive friction angle should raise ValueError."""
        with pytest.raises(ValueError, match="material_friction_angle"):
            conveyor_slope_limit(0.0, 5.0)


class TestScrewConveyorCapacity:
    """Tests for screw_conveyor_capacity."""

    def test_known_value(self):
        """Verify screw conveyor capacity formula."""
        d, p, n, f, rho = 0.3, 0.3, 60, 0.45, 1.6
        expected = math.pi / 4 * d ** 2 * p * n * f * rho * 60
        result = screw_conveyor_capacity(d, p, n, f, rho)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_rpm_higher_capacity(self):
        """Higher RPM should increase capacity."""
        q_slow = screw_conveyor_capacity(0.3, 0.3, 30, 0.4, 1.5)
        q_fast = screw_conveyor_capacity(0.3, 0.3, 90, 0.4, 1.5)
        assert q_fast > q_slow

    def test_invalid_fill_factor(self):
        """Fill factor outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="fill_factor"):
            screw_conveyor_capacity(0.3, 0.3, 60, 1.5, 1.6)

    def test_invalid_diameter(self):
        """Non-positive diameter should raise ValueError."""
        with pytest.raises(ValueError, match="diameter"):
            screw_conveyor_capacity(-0.3, 0.3, 60, 0.45, 1.6)
