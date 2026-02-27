"""Tests for minelab.equipment.pumps."""

import pytest

from minelab.equipment.pumps import (
    darcy_weisbach_friction,
    npsh_available,
    pump_head,
    pump_power,
    pump_specific_speed,
    slurry_pump_factor,
)


class TestPumpHead:
    """Tests for pump_head."""

    def test_known_value(self):
        """TDH = 30 + 1.5 + 5 = 36.5 m."""
        assert pump_head(30.0, 1.5, 5.0) == pytest.approx(36.5)

    def test_zero_losses(self):
        """Zero velocity and friction head returns static head."""
        assert pump_head(25.0, 0.0, 0.0) == pytest.approx(25.0)

    def test_negative_static_head(self):
        """Negative static head is valid (flooded suction)."""
        result = pump_head(-5.0, 1.0, 2.0)
        assert result == pytest.approx(-2.0)

    def test_invalid_velocity_head(self):
        """Negative velocity head should raise ValueError."""
        with pytest.raises(ValueError, match="velocity_head"):
            pump_head(10.0, -1.0, 3.0)

    def test_invalid_friction_head(self):
        """Negative friction head should raise ValueError."""
        with pytest.raises(ValueError, match="friction_head"):
            pump_head(10.0, 1.0, -3.0)


class TestPumpPower:
    """Tests for pump_power."""

    def test_known_value(self):
        """P = 1000 * 9.81 * 0.05 * 50 / (0.75 * 1000) = 32.7 kW."""
        result = pump_power(0.05, 50.0, 0.75)
        assert result == pytest.approx(32.7, rel=1e-3)

    def test_higher_head_more_power(self):
        """Higher head should require more power."""
        p_low = pump_power(0.05, 20.0, 0.75)
        p_high = pump_power(0.05, 80.0, 0.75)
        assert p_high > p_low

    def test_higher_efficiency_less_power(self):
        """Higher efficiency should reduce required power."""
        p_low_eff = pump_power(0.1, 50.0, 0.50)
        p_high_eff = pump_power(0.1, 50.0, 0.85)
        assert p_high_eff < p_low_eff

    def test_invalid_flow_rate(self):
        """Non-positive flow rate should raise ValueError."""
        with pytest.raises(ValueError, match="flow_rate_m3s"):
            pump_power(0.0, 50.0, 0.75)

    def test_invalid_efficiency(self):
        """Efficiency outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="efficiency"):
            pump_power(0.05, 50.0, 0.0)


class TestDarcyWeisbachFriction:
    """Tests for darcy_weisbach_friction."""

    def test_known_value(self):
        """h_f = 0.025 * (100/0.2) * 2^2 / (2*9.81) = 2.548 m."""
        g = 9.81
        expected = 0.025 * (100.0 / 0.2) * 4.0 / (2.0 * g)
        result = darcy_weisbach_friction(2.0, 0.2, 100, 0.025)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_longer_pipe_more_friction(self):
        """Longer pipe should give higher friction losses."""
        hf_short = darcy_weisbach_friction(2.0, 0.2, 50, 0.02)
        hf_long = darcy_weisbach_friction(2.0, 0.2, 200, 0.02)
        assert hf_long > hf_short

    def test_larger_diameter_less_friction(self):
        """Larger diameter should reduce friction head."""
        hf_small = darcy_weisbach_friction(2.0, 0.1, 100, 0.02)
        hf_large = darcy_weisbach_friction(2.0, 0.3, 100, 0.02)
        assert hf_large < hf_small

    def test_invalid_pipe_diameter(self):
        """Non-positive diameter should raise ValueError."""
        with pytest.raises(ValueError, match="pipe_diameter"):
            darcy_weisbach_friction(2.0, 0.0, 100, 0.025)


class TestPumpSpecificSpeed:
    """Tests for pump_specific_speed."""

    def test_known_value(self):
        """Ns = 1450 * 0.1^0.5 / 30^0.75."""
        expected = 1450.0 * 0.1 ** 0.5 / 30.0 ** 0.75
        result = pump_specific_speed(1450, 0.1, 30)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_speed_higher_ns(self):
        """Higher pump RPM should increase specific speed."""
        ns_slow = pump_specific_speed(1000, 0.1, 30)
        ns_fast = pump_specific_speed(2000, 0.1, 30)
        assert ns_fast > ns_slow

    def test_invalid_rpm(self):
        """Non-positive RPM should raise ValueError."""
        with pytest.raises(ValueError, match="n_rpm"):
            pump_specific_speed(0, 0.1, 30)

    def test_invalid_head(self):
        """Non-positive head should raise ValueError."""
        with pytest.raises(ValueError, match="h_m"):
            pump_specific_speed(1450, 0.1, 0)


class TestSlurryPumpFactor:
    """Tests for slurry_pump_factor."""

    def test_known_value(self):
        """Verify slurry SG for 30% solids concentration."""
        cw = 0.3
        denom = 1.0 - cw * (1.0 - 1.0 / 2.65)
        expected_sg = 1.0 / denom
        result = slurry_pump_factor(2.65, 1.0, 0.3)
        assert result["slurry_sg"] == pytest.approx(expected_sg, rel=1e-4)

    def test_head_ratio(self):
        """HR = 1 - 0.8 * Cw."""
        result = slurry_pump_factor(2.65, 1.0, 0.3)
        assert result["head_ratio"] == pytest.approx(0.76, rel=1e-4)

    def test_efficiency_derating(self):
        """Eff derating = 1 - 0.5 * Cw."""
        result = slurry_pump_factor(2.65, 1.0, 0.3)
        assert result["efficiency_derating"] == pytest.approx(
            0.85, rel=1e-4
        )

    def test_zero_concentration_no_derating(self):
        """Zero solids should give SG = liquid SG, HR = 1, eff = 1."""
        result = slurry_pump_factor(2.65, 1.0, 0.0)
        assert result["slurry_sg"] == pytest.approx(1.0, rel=1e-4)
        assert result["head_ratio"] == pytest.approx(1.0)
        assert result["efficiency_derating"] == pytest.approx(1.0)

    def test_invalid_sg_order(self):
        """solid_sg <= liquid_sg should raise ValueError."""
        with pytest.raises(ValueError, match="solid_sg"):
            slurry_pump_factor(1.0, 1.0, 0.3)


class TestNpshAvailable:
    """Tests for npsh_available."""

    def test_known_value(self):
        """NPSHa = 10.33 + 3.0 - 0.24 - 0.5 = 12.59."""
        result = npsh_available(10.33, 3.0, 0.24, 0.5)
        assert result == pytest.approx(12.59, rel=1e-4)

    def test_suction_lift(self):
        """Negative suction head reduces NPSHa."""
        flooded = npsh_available(10.33, 3.0, 0.24, 0.5)
        lift = npsh_available(10.33, -3.0, 0.24, 0.5)
        assert lift < flooded

    def test_high_altitude_lower_npsha(self):
        """Lower atmospheric pressure reduces NPSHa."""
        sea_level = npsh_available(10.33, 2.0, 0.24, 0.5)
        altitude = npsh_available(7.0, 2.0, 0.24, 0.5)
        assert altitude < sea_level

    def test_invalid_atmospheric_pressure(self):
        """Non-positive atmospheric pressure should raise ValueError."""
        with pytest.raises(ValueError, match="atmospheric_pressure_m"):
            npsh_available(0.0, 3.0, 0.24, 0.5)
