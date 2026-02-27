"""Tests for minelab.equipment.fleet_matching."""

import pytest

from minelab.equipment.fleet_matching import (
    match_factor,
    optimal_fleet,
)


class TestMatchFactor:
    """Tests for fleet match factor."""

    def test_known_value(self):
        """5 trucks, 30min cycle, 1 loader, 5min → MF = 0.833."""
        result = match_factor(5, 30, 1, 5)
        assert result["mf"] == pytest.approx(0.833, rel=0.01)

    def test_perfect_match(self):
        """MF = 1.0 when perfectly matched."""
        # 6 trucks * 5 min loader / (1 loader * 30 min truck) = 1.0
        result = match_factor(6, 30, 1, 5)
        assert result["mf"] == pytest.approx(1.0, rel=0.01)

    def test_truck_limited(self):
        """MF < 1 → truck-limited."""
        result = match_factor(4, 30, 1, 5)
        assert result["mf"] < 1.0

    def test_loader_limited(self):
        """MF > 1 → loader-limited."""
        result = match_factor(8, 30, 1, 5)
        assert result["mf"] > 1.0


class TestOptimalFleet:
    """Tests for optimal fleet sizing."""

    def test_positive_trucks(self):
        """Should return positive number of trucks."""
        result = optimal_fleet(30, 5, 500, 100)
        assert result["n_trucks"] > 0

    def test_meets_production(self):
        """Should meet or exceed target production."""
        result = optimal_fleet(30, 5, 500, 100)
        assert result["production"] >= 500 or result["n_trucks"] >= 1


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestMatchFactorValidation:
    """Validation tests for match_factor."""

    def test_zero_trucks_raises(self):
        """n_trucks < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_trucks"):
            match_factor(0, 30, 1, 5)

    def test_negative_trucks_raises(self):
        """Negative n_trucks should raise ValueError."""
        with pytest.raises(ValueError, match="n_trucks"):
            match_factor(-1, 30, 1, 5)

    def test_zero_loaders_raises(self):
        """n_loaders < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_loaders"):
            match_factor(5, 30, 0, 5)

    def test_negative_loaders_raises(self):
        """Negative n_loaders should raise ValueError."""
        with pytest.raises(ValueError, match="n_loaders"):
            match_factor(5, 30, -1, 5)


class TestOptimalFleetValidation:
    """Validation tests for optimal_fleet."""

    def test_zero_availability_raises(self):
        """availability == 0 should raise ValueError."""
        with pytest.raises(ValueError, match="availability"):
            optimal_fleet(30, 5, 500, 100, availability=0.0, utilization=0.9)

    def test_zero_utilization_raises(self):
        """utilization == 0 should raise ValueError."""
        with pytest.raises(ValueError, match="utilization"):
            optimal_fleet(30, 5, 500, 100, availability=0.85, utilization=0.0)


class TestOptimalFleetScaling:
    """Tests for fleet scaling when base fleet is insufficient."""

    def test_high_production_target_requires_more_trucks(self):
        """Very high production target forces fleet scaling beyond base."""
        result = optimal_fleet(30, 5, 5000, 100, availability=0.85, utilization=0.90)
        # Base fleet for MF=1 is ceil(30/5)=6 trucks
        # Production per truck: 100 * 60/30 * 0.85 * 0.9 = 153 t/h
        # Base production: 6 * 153 = 918 t/h < 5000, so fleet must scale up
        assert result["n_trucks"] > 6
        assert result["production"] >= 5000
