"""Tests for minelab.underground_mining.sublevel_methods."""

import numpy as np
import pytest

from minelab.underground_mining.sublevel_methods import (
    block_cave_draw_rate,
    draw_ellipsoid,
    ring_blast_design,
    sublevel_interval,
    sublevel_recovery,
)


class TestSublevelInterval:
    """Tests for Janelid & Kvapil (1966) sublevel interval."""

    def test_known_value(self):
        """SI = burden * (1/tan(draw) + 1/tan(dip))."""
        dip = 70.0
        draw = 60.0
        burden = 3.0
        expected = burden * (1.0 / np.tan(np.radians(draw)) + 1.0 / np.tan(np.radians(dip)))
        result = sublevel_interval(dip, draw, burden)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_steeper_dip_shorter_interval(self):
        """Steeper dip -> shorter SI."""
        si_60 = sublevel_interval(60.0, 50.0, 3.0)
        si_80 = sublevel_interval(80.0, 50.0, 3.0)
        assert si_80 < si_60

    def test_wider_draw_shorter_interval(self):
        """Wider draw angle -> shorter SI (steeper cone)."""
        si_narrow = sublevel_interval(70.0, 40.0, 3.0)
        si_wide = sublevel_interval(70.0, 70.0, 3.0)
        assert si_wide < si_narrow

    def test_larger_burden_larger_interval(self):
        """SI proportional to burden."""
        si_small = sublevel_interval(70.0, 60.0, 2.0)
        si_large = sublevel_interval(70.0, 60.0, 4.0)
        assert si_large == pytest.approx(2.0 * si_small, rel=1e-4)

    def test_invalid_dip(self):
        """Dip = 0 should raise."""
        with pytest.raises(ValueError, match="ore_dip"):
            sublevel_interval(0, 60.0, 3.0)

    def test_invalid_draw_angle(self):
        """Draw angle = 90 should raise."""
        with pytest.raises(ValueError, match="draw_angle"):
            sublevel_interval(70.0, 90.0, 3.0)


class TestDrawEllipsoid:
    """Tests for Kvapil (1982) draw ellipsoid."""

    def test_semi_axes(self):
        """height=30, draw_angle=60 -> a=15, b=15*tan(60)."""
        result = draw_ellipsoid(30.0, 60.0)
        assert result["semi_major_m"] == pytest.approx(15.0, rel=1e-4)
        expected_b = 30.0 * np.tan(np.radians(60.0)) / 2.0
        assert result["semi_minor_m"] == pytest.approx(expected_b, rel=1e-4)

    def test_volume_formula(self):
        """Volume = (4/3)*pi*a*b^2."""
        result = draw_ellipsoid(30.0, 45.0)
        a = result["semi_major_m"]
        b = result["semi_minor_m"]
        expected_vol = (4.0 / 3.0) * np.pi * a * b * b
        assert result["volume_m3"] == pytest.approx(expected_vol, rel=1e-4)

    def test_eccentricity_range(self):
        """Eccentricity should be in [0, 1)."""
        result = draw_ellipsoid(30.0, 30.0)
        assert 0 <= result["eccentricity"] < 1.0

    def test_wider_angle_larger_volume(self):
        """Wider draw angle -> larger volume."""
        narrow = draw_ellipsoid(30.0, 30.0)["volume_m3"]
        wide = draw_ellipsoid(30.0, 60.0)["volume_m3"]
        assert wide > narrow

    def test_invalid_height(self):
        """Zero height should raise."""
        with pytest.raises(ValueError, match="height"):
            draw_ellipsoid(0, 60.0)


class TestSublevelRecovery:
    """Tests for Laubscher (1994) recovery and dilution."""

    def test_full_draw(self):
        """draw_height >= SI -> recovery near max (0.85)."""
        result = sublevel_recovery(30.0, 30.0, 3.0, 2.7)
        assert result["recovery_fraction"] == pytest.approx(0.85, rel=1e-4)

    def test_partial_draw(self):
        """draw_height < SI -> lower recovery."""
        result = sublevel_recovery(15.0, 30.0, 3.0, 2.7)
        assert result["recovery_fraction"] < 0.85

    def test_dilution_non_negative(self):
        """Dilution should be non-negative."""
        result = sublevel_recovery(20.0, 25.0, 3.0, 2.7)
        assert result["dilution_fraction"] >= 0

    def test_over_draw_recovery_capped(self):
        """Over-drawing should not exceed 1.0 recovery."""
        result = sublevel_recovery(50.0, 30.0, 3.0, 2.7)
        assert result["recovery_fraction"] <= 1.0

    def test_invalid_draw_height(self):
        """Negative draw height should raise."""
        with pytest.raises(ValueError, match="draw_height"):
            sublevel_recovery(-1.0, 30.0, 3.0, 2.7)


class TestRingBlastDesign:
    """Tests for Hustrulid & Bullock ring blast design."""

    def test_area_per_ring(self):
        """area = burden * toe_spacing."""
        result = ring_blast_design(0.089, 2.5, 3.0)
        assert result["area_per_ring_m2"] == pytest.approx(7.5, rel=1e-4)

    def test_pattern_ratio(self):
        """pattern_ratio = toe_spacing / burden."""
        result = ring_blast_design(0.089, 2.5, 3.0)
        assert result["pattern_ratio"] == pytest.approx(3.0 / 2.5, rel=1e-4)

    def test_drill_metres_positive(self):
        """Drill metres should be positive."""
        result = ring_blast_design(0.089, 2.5, 3.0)
        assert result["drill_metres_per_ring"] > 0

    def test_invalid_diameter(self):
        """Zero diameter should raise."""
        with pytest.raises(ValueError, match="diameter"):
            ring_blast_design(0, 2.5, 3.0)

    def test_invalid_burden(self):
        """Negative burden should raise."""
        with pytest.raises(ValueError, match="burden"):
            ring_blast_design(0.089, -1.0, 3.0)


class TestBlockCaveDrawRate:
    """Tests for Laubscher (1994) block cave production."""

    def test_known_value(self):
        """200m column, 0.5 m/d, 5000 m2, 2.7 t/m3."""
        result = block_cave_draw_rate(200.0, 0.5, 5000.0, 2.7)
        assert result["total_ore_tonnes"] == pytest.approx(2_700_000.0, rel=1e-4)
        assert result["draw_time_days"] == pytest.approx(400.0, rel=1e-4)
        expected_daily = 5000.0 * 0.5 * 2.7
        assert result["daily_production_tonnes"] == pytest.approx(expected_daily, rel=1e-4)

    def test_higher_rate_shorter_time(self):
        """Higher cave rate -> shorter draw time."""
        slow = block_cave_draw_rate(200.0, 0.3, 5000.0, 2.7)
        fast = block_cave_draw_rate(200.0, 0.8, 5000.0, 2.7)
        assert fast["draw_time_days"] < slow["draw_time_days"]

    def test_total_ore_independent_of_rate(self):
        """Total ore doesn't depend on draw rate."""
        slow = block_cave_draw_rate(200.0, 0.3, 5000.0, 2.7)
        fast = block_cave_draw_rate(200.0, 0.8, 5000.0, 2.7)
        assert slow["total_ore_tonnes"] == pytest.approx(fast["total_ore_tonnes"], rel=1e-4)

    def test_invalid_column_height(self):
        """Zero column height should raise."""
        with pytest.raises(ValueError, match="column_height"):
            block_cave_draw_rate(0, 0.5, 5000.0, 2.7)
