"""Tests for minelab.geomechanics.support_design."""

import pytest

from minelab.geomechanics.support_design import (
    pillar_strength_bieniawski,
    pillar_strength_lunder_pakalnis,
    rock_bolt_design,
    shotcrete_thickness,
    stand_up_time,
    tributary_area_stress,
)


class TestPillarStrengthBieniawski:
    """Tests for Bieniawski pillar strength formula."""

    def test_known_value(self):
        """w=10, h=5, σci=100 → σp = 100*(0.64+0.36*2) = 136."""
        result = pillar_strength_bieniawski(10, 5, 100)
        assert result["strength"] == pytest.approx(136.0, rel=0.01)
        assert result["w_over_h"] == 2.0

    def test_square_pillar(self):
        """w/h=1 → σp = σci*(0.64+0.36) = σci."""
        result = pillar_strength_bieniawski(5, 5, 100)
        assert result["strength"] == pytest.approx(100.0, rel=0.01)

    def test_higher_ratio_stronger(self):
        """Higher w/h → stronger pillar."""
        narrow = pillar_strength_bieniawski(5, 5, 100)["strength"]
        wide = pillar_strength_bieniawski(10, 5, 100)["strength"]
        assert wide > narrow

    def test_invalid_width(self):
        """Negative width should raise."""
        with pytest.raises(ValueError):
            pillar_strength_bieniawski(-5, 5, 100)


class TestPillarStrengthLunderPakalnis:
    """Tests for Lunder-Pakalnis pillar strength."""

    def test_positive_strength(self):
        """Should return positive strength."""
        result = pillar_strength_lunder_pakalnis(10, 5, 100)
        assert result["strength"] > 0

    def test_wider_pillar_stronger(self):
        """Higher w/h → stronger pillar."""
        narrow = pillar_strength_lunder_pakalnis(5, 5, 100)["strength"]
        wide = pillar_strength_lunder_pakalnis(10, 5, 100)["strength"]
        assert wide > narrow

    def test_kappa_exists(self):
        """Kappa parameter should be computed."""
        result = pillar_strength_lunder_pakalnis(10, 5, 100)
        assert "kappa" in result


class TestTributaryAreaStress:
    """Tests for tributary area stress calculation."""

    def test_known_value(self):
        """depth=100, e=0.75, ρ=2700 → σp ≈ 10.59 MPa."""
        result = tributary_area_stress(100, 0.75, 2700)
        assert result["pillar_stress"] == pytest.approx(10.59, rel=0.01)

    def test_stress_concentration(self):
        """e=0.75 → SCF = 4."""
        result = tributary_area_stress(100, 0.75, 2700)
        assert result["stress_concentration"] == pytest.approx(4.0, rel=0.01)

    def test_deeper_higher_stress(self):
        """Deeper mining → higher pillar stress."""
        shallow = tributary_area_stress(100, 0.5)["pillar_stress"]
        deep = tributary_area_stress(500, 0.5)["pillar_stress"]
        assert deep > shallow

    def test_higher_extraction_higher_stress(self):
        """Higher extraction ratio → higher pillar stress."""
        low_e = tributary_area_stress(100, 0.5)["pillar_stress"]
        high_e = tributary_area_stress(100, 0.8)["pillar_stress"]
        assert high_e > low_e

    def test_invalid_extraction(self):
        """e >= 1 should raise."""
        with pytest.raises(ValueError):
            tributary_area_stress(100, 1.0)


class TestRockBoltDesign:
    """Tests for rock bolt design from Q-system."""

    def test_known_case(self):
        """Q=10, span=10m → reasonable bolt length."""
        result = rock_bolt_design(10, 10, 1.0)
        assert result["bolt_length"] == pytest.approx(3.5, rel=0.01)
        assert result["spacing"] > 0

    def test_longer_for_wider_span(self):
        """Wider span → longer bolts."""
        narrow = rock_bolt_design(10, 5)["bolt_length"]
        wide = rock_bolt_design(10, 15)["bolt_length"]
        assert wide > narrow

    def test_equivalent_dimension(self):
        """De = span/ESR."""
        result = rock_bolt_design(10, 10, 2.0)
        assert result["equivalent_dimension"] == pytest.approx(5.0)


class TestShotcreteThickness:
    """Tests for shotcrete thickness estimation."""

    def test_good_rock_no_support(self):
        """RMR >= 81 → 0 mm shotcrete."""
        result = shotcrete_thickness(85, 5)
        assert result["thickness"] == 0.0

    def test_fair_rock(self):
        """RMR 41-60 → positive thickness."""
        result = shotcrete_thickness(50, 5)
        assert result["thickness"] > 0

    def test_poor_rock_thicker(self):
        """Lower RMR → thicker shotcrete."""
        fair = shotcrete_thickness(50, 5)["thickness"]
        poor = shotcrete_thickness(30, 5)["thickness"]
        assert poor > fair

    def test_invalid_rmr(self):
        """RMR > 100 should raise."""
        with pytest.raises(ValueError):
            shotcrete_thickness(110, 5)


class TestStandUpTime:
    """Tests for stand-up time estimation."""

    def test_positive_time(self):
        """Should return positive time."""
        result = stand_up_time(60, 5)
        assert result["time_hours"] > 0
        assert result["time_days"] > 0

    def test_higher_rmr_longer(self):
        """Higher RMR → longer stand-up time."""
        low = stand_up_time(30, 5)["time_hours"]
        high = stand_up_time(70, 5)["time_hours"]
        assert high > low

    def test_wider_span_shorter(self):
        """Wider span → shorter stand-up time."""
        narrow = stand_up_time(60, 3)["time_hours"]
        wide = stand_up_time(60, 10)["time_hours"]
        assert wide < narrow

    def test_days_conversion(self):
        """time_days = time_hours / 24."""
        result = stand_up_time(60, 5)
        assert result["time_days"] == pytest.approx(
            result["time_hours"] / 24, rel=1e-6
        )
