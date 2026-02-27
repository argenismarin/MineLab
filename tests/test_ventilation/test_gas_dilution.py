"""Tests for minelab.ventilation.gas_dilution."""

import pytest

from minelab.ventilation.gas_dilution import (
    air_for_blasting,
    air_for_diesel,
    dust_dilution,
    methane_dilution,
)


class TestAirForDiesel:
    """Tests for diesel air requirement."""

    def test_known_value(self):
        """200 kW → 12 m³/s at sea level."""
        q = air_for_diesel(200)
        assert q == pytest.approx(12.0, rel=0.01)

    def test_higher_altitude_more_air(self):
        """Higher altitude → more air needed."""
        q_sea = air_for_diesel(200, altitude=0)
        q_high = air_for_diesel(200, altitude=3000)
        assert q_high > q_sea


class TestAirForBlasting:
    """Tests for blasting air requirement."""

    def test_positive(self):
        """Should return positive Q."""
        q = air_for_blasting(100, 600)
        assert q > 0

    def test_more_powder_more_air(self):
        """More explosives → more air."""
        q_small = air_for_blasting(50, 600)
        q_large = air_for_blasting(200, 600)
        assert q_large > q_small


class TestMethaneDilution:
    """Tests for methane dilution."""

    def test_known_value(self):
        """0.5 m³/s at 1% → 50 m³/s."""
        q = methane_dilution(0.5, 0.01)
        assert q == pytest.approx(50.0, rel=0.01)


class TestDustDilution:
    """Tests for dust dilution."""

    def test_positive(self):
        """Should return positive Q."""
        q = dust_dilution(5.0, 1.0)
        assert q > 0

    def test_known(self):
        """5 mg/s at TLV=1 mg/m³ → 5 m³/s."""
        q = dust_dilution(5.0, 1.0)
        assert q == pytest.approx(5.0, rel=0.01)
