"""Tests for minelab.mine_planning.underground_planning."""

import numpy as np
import pytest

from minelab.mine_planning.underground_planning import (
    crown_pillar_thickness,
    development_advance_rate,
    long_hole_production_rate,
    mining_recovery_underground,
    stope_economic_value,
    underground_cutoff_grade,
)


class TestStopeEconomicValue:
    """Tests for NSR-based stope economic value."""

    def test_profitable_stope(self):
        """10000t, 5 g/t, $60/g, 90% rec, $80/t opex, 10% dil."""
        result = stope_economic_value(
            10000.0,
            5.0,
            60.0,
            0.9,
            80.0,
            0.1,
        )
        diluted_grade = 5.0 * 0.9  # 4.5
        revenue = 10000.0 * diluted_grade * 60.0 * 0.9
        cost = 10000.0 * 80.0
        assert result["diluted_grade"] == pytest.approx(4.5, rel=1e-4)
        assert result["revenue"] == pytest.approx(revenue, rel=1e-4)
        assert result["cost"] == pytest.approx(cost, rel=1e-4)
        assert result["profit"] == pytest.approx(revenue - cost, rel=1e-4)

    def test_unprofitable_stope(self):
        """Low grade -> negative profit."""
        result = stope_economic_value(
            10000.0,
            1.0,
            60.0,
            0.9,
            80.0,
            0.1,
        )
        assert result["profit"] < 0

    def test_profit_per_tonne(self):
        """profit_per_tonne = profit / ore_tonnes."""
        result = stope_economic_value(
            5000.0,
            5.0,
            60.0,
            0.9,
            80.0,
            0.1,
        )
        assert result["profit_per_tonne"] == pytest.approx(result["profit"] / 5000.0, rel=1e-4)

    def test_higher_dilution_less_profit(self):
        """More dilution -> less profit."""
        low_dil = stope_economic_value(
            10000.0,
            5.0,
            60.0,
            0.9,
            80.0,
            0.05,
        )
        high_dil = stope_economic_value(
            10000.0,
            5.0,
            60.0,
            0.9,
            80.0,
            0.3,
        )
        assert low_dil["profit"] > high_dil["profit"]

    def test_invalid_recovery(self):
        """Recovery > 1 should raise."""
        with pytest.raises(ValueError, match="recovery"):
            stope_economic_value(
                10000.0,
                5.0,
                60.0,
                1.5,
                80.0,
                0.1,
            )

    def test_invalid_dilution(self):
        """Dilution > 1 should raise."""
        with pytest.raises(ValueError, match="dilution"):
            stope_economic_value(
                10000.0,
                5.0,
                60.0,
                0.9,
                80.0,
                1.5,
            )


class TestUndergroundCutoffGrade:
    """Tests for Lane (1988) underground cut-off grade."""

    def test_known_value(self):
        """COG = (30 + 50) / (60 * 0.9) = 1.4815."""
        result = underground_cutoff_grade(30.0, 60.0, 0.9, 50.0)
        expected = (30.0 + 50.0) / (60.0 * 0.9)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_costs_higher_cog(self):
        """Higher costs -> higher COG."""
        low = underground_cutoff_grade(20.0, 60.0, 0.9, 30.0)
        high = underground_cutoff_grade(40.0, 60.0, 0.9, 60.0)
        assert high > low

    def test_higher_price_lower_cog(self):
        """Higher metal price -> lower COG."""
        low_price = underground_cutoff_grade(30.0, 40.0, 0.9, 50.0)
        high_price = underground_cutoff_grade(30.0, 100.0, 0.9, 50.0)
        assert high_price < low_price

    def test_higher_recovery_lower_cog(self):
        """Higher recovery -> lower COG."""
        low_rec = underground_cutoff_grade(30.0, 60.0, 0.7, 50.0)
        high_rec = underground_cutoff_grade(30.0, 60.0, 0.95, 50.0)
        assert high_rec < low_rec

    def test_invalid_price(self):
        """Zero price should raise."""
        with pytest.raises(ValueError, match="price"):
            underground_cutoff_grade(30.0, 0.0, 0.9, 50.0)


class TestMiningRecoveryUnderground:
    """Tests for mining recovery and dilution estimation."""

    def test_known_value(self):
        """ore_w=4, skin=0.5, factor=0.95."""
        result = mining_recovery_underground(5.0, 4.0, 0.5, 0.95)
        effective = 4.0 + 2.0 * 0.5
        expected_rec = (4.0 / effective) * 0.95
        assert result["mining_recovery"] == pytest.approx(expected_rec, rel=1e-4)

    def test_dilution_complement(self):
        """dilution = 1 - recovery."""
        result = mining_recovery_underground(5.0, 4.0, 0.3, 0.9)
        assert result["dilution"] == pytest.approx(1.0 - result["mining_recovery"], rel=1e-4)

    def test_effective_width(self):
        """effective_width = ore_width + 2*skin."""
        result = mining_recovery_underground(5.0, 4.0, 0.5, 0.95)
        assert result["effective_width"] == pytest.approx(5.0, rel=1e-4)

    def test_no_dilution_skin(self):
        """Zero skin -> recovery = method_factor."""
        result = mining_recovery_underground(4.0, 4.0, 0.0, 0.95)
        assert result["mining_recovery"] == pytest.approx(0.95, rel=1e-4)

    def test_invalid_method_factor(self):
        """Factor > 1 should raise."""
        with pytest.raises(ValueError, match="mining_method_factor"):
            mining_recovery_underground(5.0, 4.0, 0.5, 1.5)


class TestLongHoleProductionRate:
    """Tests for long-hole stoping production rate."""

    def test_basic(self):
        """8 holes, 2.5m burden, 20 m/h drill, 2h charge, 1 day."""
        result = long_hole_production_rate(8, 2.5, 20.0, 2.0, 1.0)
        ring_depth = 1.1 * 2.5
        expected_drill = 8 * ring_depth / 20.0
        assert result["drill_time_hours"] == pytest.approx(expected_drill, rel=1e-4)

    def test_tonnes_per_blast(self):
        """tonnes = burden^2 * holes * 2.7."""
        result = long_hole_production_rate(8, 2.5, 20.0, 2.0, 1.0)
        expected = 2.5 * 2.5 * 8 * 2.7
        assert result["tonnes_per_blast"] == pytest.approx(expected, rel=1e-4)

    def test_longer_interval_lower_daily(self):
        """Longer blast interval -> lower daily production."""
        freq = long_hole_production_rate(8, 2.5, 20.0, 2.0, 1.0)
        infreq = long_hole_production_rate(8, 2.5, 20.0, 2.0, 3.0)
        assert infreq["daily_production_tonnes"] < (freq["daily_production_tonnes"])

    def test_invalid_drill_rate(self):
        """Zero drill rate should raise."""
        with pytest.raises(ValueError, match="drill_rate"):
            long_hole_production_rate(8, 2.5, 0.0, 2.0, 1.0)


class TestDevelopmentAdvanceRate:
    """Tests for development advance rate."""

    def test_basic(self):
        """16 m2, 2 rounds/day, 3.5 m/round."""
        result = development_advance_rate(16.0, 2.0, 3.5)
        assert result["daily_advance_m"] == pytest.approx(7.0, rel=1e-4)
        assert result["monthly_advance_m"] == pytest.approx(175.0, rel=1e-4)
        assert result["daily_volume_m3"] == pytest.approx(112.0, rel=1e-4)

    def test_more_rounds_faster(self):
        """More rounds -> faster advance."""
        slow = development_advance_rate(16.0, 1.0, 3.5)
        fast = development_advance_rate(16.0, 3.0, 3.5)
        assert fast["daily_advance_m"] > slow["daily_advance_m"]

    def test_monthly_is_25_days(self):
        """Monthly = daily * 25."""
        result = development_advance_rate(16.0, 2.0, 3.5)
        assert result["monthly_advance_m"] == pytest.approx(
            result["daily_advance_m"] * 25.0, rel=1e-4
        )

    def test_invalid_area(self):
        """Zero area should raise."""
        with pytest.raises(ValueError, match="drill_pattern_area"):
            development_advance_rate(0, 2.0, 3.5)


class TestCrownPillarThickness:
    """Tests for Carter (1992) crown pillar design."""

    def test_basic(self):
        """span=15, density=2700, sigma_cm=20, SF=2."""
        gamma = 2700.0 * 9.81 / 1e6
        expected = 15.0 * np.sqrt(gamma / 20.0) * 2.0
        result = crown_pillar_thickness(15.0, 2700.0, 20.0, 2.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_wider_span_thicker(self):
        """Wider span -> thicker crown pillar."""
        narrow = crown_pillar_thickness(10.0, 2700.0, 20.0, 2.0)
        wide = crown_pillar_thickness(25.0, 2700.0, 20.0, 2.0)
        assert wide > narrow

    def test_stronger_rock_thinner(self):
        """Stronger rock -> thinner crown pillar."""
        weak = crown_pillar_thickness(15.0, 2700.0, 10.0, 2.0)
        strong = crown_pillar_thickness(15.0, 2700.0, 40.0, 2.0)
        assert strong < weak

    def test_higher_sf_thicker(self):
        """Higher safety factor -> thicker pillar."""
        sf_low = crown_pillar_thickness(15.0, 2700.0, 20.0, 1.5)
        sf_high = crown_pillar_thickness(15.0, 2700.0, 20.0, 3.0)
        assert sf_high > sf_low

    def test_invalid_span(self):
        """Zero span should raise."""
        with pytest.raises(ValueError, match="span"):
            crown_pillar_thickness(0, 2700.0, 20.0, 2.0)

    def test_invalid_sigma_cm(self):
        """Zero sigma_cm should raise."""
        with pytest.raises(ValueError, match="sigma_cm"):
            crown_pillar_thickness(15.0, 2700.0, 0.0, 2.0)
