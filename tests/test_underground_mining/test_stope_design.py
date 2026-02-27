"""Tests for minelab.underground_mining.stope_design."""

import pytest

from minelab.underground_mining.stope_design import (
    hydraulic_radius,
    mathews_stability,
    mucking_rate,
    rill_angle,
    stope_dimensions,
    undercut_design,
)


class TestMathewsStability:
    """Tests for Potvin (1988) stability number."""

    def test_known_stable(self):
        """Q'=10, A=0.8, B=0.5, C=4 -> N'=16, stable."""
        result = mathews_stability(10.0, 0.8, 0.5, 4.0)
        assert result["n_prime"] == pytest.approx(16.0, rel=1e-4)
        assert result["stability_zone"] == "stable"

    def test_transition_zone(self):
        """N' between 0.1 and 4 -> transition."""
        result = mathews_stability(1.0, 0.5, 0.3, 2.0)
        assert result["n_prime"] == pytest.approx(0.3, rel=1e-4)
        assert result["stability_zone"] == "transition"

    def test_unstable_zone(self):
        """Very low N' -> unstable."""
        result = mathews_stability(0.1, 0.2, 0.2, 1.0)
        assert result["n_prime"] == pytest.approx(0.004, rel=1e-4)
        assert result["stability_zone"] == "unstable"

    def test_hr_limit_positive(self):
        """HR limit should be positive for stable stope."""
        result = mathews_stability(10.0, 1.0, 1.0, 5.0)
        assert result["hydraulic_radius_limit"] > 0

    def test_invalid_q_prime(self):
        """Negative Q' should raise."""
        with pytest.raises(ValueError, match="q_prime"):
            mathews_stability(-1.0, 0.5, 0.5, 2.0)

    def test_invalid_a_range(self):
        """A > 1 should raise."""
        with pytest.raises(ValueError, match="a"):
            mathews_stability(5.0, 1.5, 0.5, 2.0)

    def test_invalid_b_range(self):
        """B < 0.2 should raise."""
        with pytest.raises(ValueError, match="b"):
            mathews_stability(5.0, 0.5, 0.1, 2.0)


class TestHydraulicRadius:
    """Tests for HR = Area / Perimeter."""

    def test_square(self):
        """10x10 -> HR = 100/40 = 2.5."""
        assert hydraulic_radius(10.0, 10.0) == pytest.approx(2.5, rel=1e-4)

    def test_rectangular(self):
        """10x5 -> HR = 50/30 = 1.667."""
        assert hydraulic_radius(10.0, 5.0) == pytest.approx(50.0 / 30.0, rel=1e-4)

    def test_narrow(self):
        """100x1 -> HR = 100/202 = 0.4950."""
        assert hydraulic_radius(100.0, 1.0) == pytest.approx(100.0 / 202.0, rel=1e-4)

    def test_invalid_length(self):
        """Zero length should raise."""
        with pytest.raises(ValueError, match="length"):
            hydraulic_radius(0, 5.0)

    def test_invalid_width(self):
        """Negative width should raise."""
        with pytest.raises(ValueError, match="width"):
            hydraulic_radius(10.0, -1.0)


class TestStopeDimensions:
    """Tests for max strike length from HR constraint."""

    def test_basic_constraint(self):
        """Verify max strike length satisfies HR limit."""
        result = stope_dimensions(5.0, 70.0, 30.0, 8.0)
        assert result["actual_hr"] == pytest.approx(8.0, rel=0.01)
        assert result["max_strike_length"] > 0

    def test_stope_volume(self):
        """Volume = strike * height * width."""
        result = stope_dimensions(5.0, 70.0, 30.0, 8.0)
        expected_vol = result["max_strike_length"] * 30.0 * 5.0
        assert result["stope_volume"] == pytest.approx(expected_vol, rel=1e-4)

    def test_unconstrained_case(self):
        """When hr_limit > height/2, effectively unconstrained."""
        result = stope_dimensions(5.0, 70.0, 20.0, 15.0)
        assert result["max_strike_length"] >= 100.0

    def test_invalid_dip(self):
        """Dip > 90 should raise."""
        with pytest.raises(ValueError, match="dip"):
            stope_dimensions(5.0, 95.0, 30.0, 8.0)


class TestRillAngle:
    """Tests for effective rill angle."""

    def test_vertical_ore(self):
        """Dip=90 -> rill = repose angle."""
        result = rill_angle(37.0, 90.0)
        assert result == pytest.approx(37.0, rel=1e-4)

    def test_zero_dip(self):
        """Dip=0 -> rill = 0 (horizontal ore, no rill)."""
        result = rill_angle(37.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_moderate_dip(self):
        """Intermediate dip gives angle between 0 and repose."""
        result = rill_angle(37.0, 60.0)
        assert 0.0 < result < 37.0

    def test_monotonic_with_dip(self):
        """Higher dip -> higher rill angle."""
        low = rill_angle(37.0, 30.0)
        high = rill_angle(37.0, 70.0)
        assert high > low

    def test_invalid_repose(self):
        """Repose > 90 should raise."""
        with pytest.raises(ValueError, match="repose_angle"):
            rill_angle(95.0, 60.0)


class TestUndercutDesign:
    """Tests for undercut ring blast geometry."""

    def test_basic_design(self):
        """Standard 89mm hole, 10m width."""
        result = undercut_design(10.0, 0.089, 0.5)
        assert result["toe_spacing"] == pytest.approx(28.0 * 0.089, rel=1e-4)
        assert result["burden"] == pytest.approx(25.0 * 0.089, rel=1e-4)
        assert result["holes_per_ring"] >= 1
        assert result["explosive_per_ring_kg"] > 0

    def test_explosive_amount(self):
        """Explosive = burden * width * 1.0 * PF."""
        result = undercut_design(10.0, 0.089, 0.5)
        expected = result["burden"] * 10.0 * 1.0 * 0.5
        assert result["explosive_per_ring_kg"] == pytest.approx(expected, rel=1e-4)

    def test_wider_ore_more_holes(self):
        """Wider ore body -> more holes per ring."""
        narrow = undercut_design(5.0, 0.089, 0.5)["holes_per_ring"]
        wide = undercut_design(20.0, 0.089, 0.5)["holes_per_ring"]
        assert wide >= narrow

    def test_invalid_diameter(self):
        """Zero diameter should raise."""
        with pytest.raises(ValueError, match="blast_hole_diam"):
            undercut_design(10.0, 0, 0.5)


class TestMuckingRate:
    """Tests for LHD mucking productivity."""

    def test_known_value(self):
        """6 m3 bucket, 0.85 FF, 5 min cycle, 2.7 t/m3."""
        result = mucking_rate(6.0, 0.85, 5.0, 2.7)
        expected = 6.0 * 0.85 * 2.7 * 60.0 / 5.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_capacity_more_production(self):
        """Bigger bucket -> higher rate."""
        small = mucking_rate(4.0, 0.85, 5.0, 2.7)
        large = mucking_rate(8.0, 0.85, 5.0, 2.7)
        assert large > small

    def test_longer_cycle_less_production(self):
        """Longer cycle -> lower rate."""
        fast = mucking_rate(6.0, 0.85, 3.0, 2.7)
        slow = mucking_rate(6.0, 0.85, 10.0, 2.7)
        assert fast > slow

    def test_invalid_fill_factor(self):
        """Fill factor > 1 should raise."""
        with pytest.raises(ValueError, match="fill_factor"):
            mucking_rate(6.0, 1.5, 5.0, 2.7)

    def test_invalid_cycle_time(self):
        """Negative cycle time should raise."""
        with pytest.raises(ValueError, match="cycle_time_min"):
            mucking_rate(6.0, 0.85, -1.0, 2.7)
