"""Tests for minelab.environmental.water_balance."""

import pytest

from minelab.environmental.water_balance import (
    pit_dewatering_estimate,
    runoff_coefficient,
    site_water_balance,
)


class TestSiteWaterBalance:
    """Tests for site water balance."""

    def test_positive_inflow(self):
        """Net positive inflow → storage increases."""
        result = site_water_balance([100], [50], [20], [10])
        assert result["final_storage"] > 0

    def test_multiple_periods(self):
        """Multiple periods → cumulative storage tracked."""
        result = site_water_balance(
            [100, 80, 120], [50, 60, 40], [10, 10, 10], [20, 20, 20]
        )
        assert len(result["cumulative_storage"]) == 3


class TestPitDewatering:
    """Tests for pit dewatering estimate."""

    def test_known_value(self):
        """K=1e-5, head=0.1, A=1000 → Q=0.001."""
        q = pit_dewatering_estimate(1e-5, 0.1, 1000)
        assert q == pytest.approx(0.001, rel=0.05)

    def test_positive(self):
        """Flow should be positive."""
        q = pit_dewatering_estimate(1e-4, 0.5, 500)
        assert q > 0


class TestRunoffCoefficient:
    """Tests for rational method runoff."""

    def test_positive(self):
        """Should return positive runoff."""
        q = runoff_coefficient(50, 10000, 0.8)
        assert q > 0

    def test_higher_rain_more_runoff(self):
        """More rain → more runoff."""
        q_low = runoff_coefficient(10, 10000, 0.8)
        q_high = runoff_coefficient(50, 10000, 0.8)
        assert q_high > q_low


class TestSiteWaterBalanceValidation:
    """Tests for site water balance input validation."""

    def test_mismatched_lengths_raises(self):
        """Mismatched list lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            site_water_balance([100, 50], [30], [10, 10], [20, 20])

    def test_empty_lists_raises(self):
        """Empty input lists should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            site_water_balance([], [], [], [])

    def test_negative_precipitation_raises(self):
        """Negative precipitation values should raise ValueError."""
        with pytest.raises(ValueError, match="precipitation"):
            site_water_balance([-10, 50], [30, 20], [10, 10], [20, 15])

    def test_negative_evaporation_raises(self):
        """Negative evaporation values should raise ValueError."""
        with pytest.raises(ValueError, match="evaporation"):
            site_water_balance([100, 50], [-30, 20], [10, 10], [20, 15])

    def test_negative_inflow_raises(self):
        """Negative inflow values should raise ValueError."""
        with pytest.raises(ValueError, match="inflow"):
            site_water_balance([100, 50], [30, 20], [-10, 10], [20, 15])

    def test_negative_outflow_raises(self):
        """Negative outflow values should raise ValueError."""
        with pytest.raises(ValueError, match="outflow"):
            site_water_balance([100, 50], [30, 20], [10, 10], [-20, 15])
