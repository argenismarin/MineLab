"""Tests for minelab.environmental.carbon."""

import pytest

from minelab.environmental.carbon import (
    blasting_emissions,
    carbon_intensity,
    diesel_emissions,
    scope1_scope2_emissions,
)


# ---------------------------------------------------------------------------
# diesel_emissions
# ---------------------------------------------------------------------------


class TestDieselEmissions:
    """Tests for diesel_emissions."""

    def test_known_value_default_ef(self):
        """1000 L * 2.68 = 2680 kg CO2."""
        result = diesel_emissions(1000)
        assert result["co2_kg"] == pytest.approx(2680.0, rel=1e-4)
        assert result["co2_tonnes"] == pytest.approx(2.68, rel=1e-4)

    def test_custom_emission_factor(self):
        """500 L * 3.0 = 1500 kg CO2."""
        result = diesel_emissions(500, 3.0)
        assert result["co2_kg"] == pytest.approx(1500.0, rel=1e-4)
        assert result["co2_tonnes"] == pytest.approx(1.5, rel=1e-4)

    def test_zero_litres(self):
        """Zero litres -> zero emissions."""
        result = diesel_emissions(0)
        assert result["co2_kg"] == pytest.approx(0.0)
        assert result["co2_tonnes"] == pytest.approx(0.0)

    def test_tonnes_equals_kg_over_1000(self):
        """co2_tonnes should equal co2_kg / 1000."""
        result = diesel_emissions(7500)
        assert result["co2_tonnes"] == pytest.approx(
            result["co2_kg"] / 1000.0, rel=1e-6
        )

    def test_invalid_litres(self):
        """Negative litres raises ValueError."""
        with pytest.raises(ValueError, match="diesel_litres"):
            diesel_emissions(-100)

    def test_invalid_emission_factor(self):
        """Zero emission factor raises ValueError."""
        with pytest.raises(
            ValueError, match="emission_factor_kgco2_per_litre"
        ):
            diesel_emissions(1000, 0)


# ---------------------------------------------------------------------------
# blasting_emissions
# ---------------------------------------------------------------------------


class TestBlastingEmissions:
    """Tests for blasting_emissions."""

    def test_known_value(self):
        """1000 kg ANFO + 500 kg emulsion = 170 + 75 = 245 kg CO2."""
        result = blasting_emissions(1000, 500)
        assert result == pytest.approx(245.0, rel=1e-4)

    def test_anfo_only(self):
        """ANFO only: 2000 * 0.17 = 340 kg."""
        result = blasting_emissions(2000, 0)
        assert result == pytest.approx(340.0, rel=1e-4)

    def test_emulsion_only(self):
        """Emulsion only: 3000 * 0.15 = 450 kg."""
        result = blasting_emissions(0, 3000)
        assert result == pytest.approx(450.0, rel=1e-4)

    def test_zero_both(self):
        """Zero explosives -> zero emissions."""
        result = blasting_emissions(0, 0)
        assert result == pytest.approx(0.0)

    def test_invalid_anfo(self):
        """Negative ANFO raises ValueError."""
        with pytest.raises(ValueError, match="anfo_kg"):
            blasting_emissions(-100, 500)

    def test_invalid_emulsion(self):
        """Negative emulsion raises ValueError."""
        with pytest.raises(ValueError, match="emulsion_kg"):
            blasting_emissions(100, -200)


# ---------------------------------------------------------------------------
# carbon_intensity
# ---------------------------------------------------------------------------


class TestCarbonIntensity:
    """Tests for carbon_intensity."""

    def test_known_value(self):
        """50000 t CO2 / 10000 t metal = 5.0."""
        result = carbon_intensity(50_000, 10_000)
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_zero_emissions(self):
        """Zero GHG -> zero intensity."""
        result = carbon_intensity(0, 10_000)
        assert result == pytest.approx(0.0)

    def test_high_intensity(self):
        """Low production with high emissions -> high CI."""
        result = carbon_intensity(100_000, 500)
        assert result == pytest.approx(200.0, rel=1e-4)

    def test_invalid_production(self):
        """Zero production raises ValueError."""
        with pytest.raises(
            ValueError, match="annual_production_t_metal"
        ):
            carbon_intensity(50_000, 0)

    def test_invalid_ghg(self):
        """Negative GHG raises ValueError."""
        with pytest.raises(ValueError, match="total_ghg_t_co2eq"):
            carbon_intensity(-100, 10_000)


# ---------------------------------------------------------------------------
# scope1_scope2_emissions
# ---------------------------------------------------------------------------


class TestScope1Scope2Emissions:
    """Tests for scope1_scope2_emissions."""

    def test_known_value(self):
        """Scope1=500, Scope2=1e6*0.5/1000=500, total=1000."""
        result = scope1_scope2_emissions(500, 1_000_000, 0.5)
        assert result["scope1_tco2"] == pytest.approx(500.0, rel=1e-4)
        assert result["scope2_tco2"] == pytest.approx(500.0, rel=1e-4)
        assert result["total_tco2"] == pytest.approx(1000.0, rel=1e-4)

    def test_zero_electricity(self):
        """No electricity -> Scope 2 = 0."""
        result = scope1_scope2_emissions(300, 0, 0.8)
        assert result["scope2_tco2"] == pytest.approx(0.0)
        assert result["total_tco2"] == pytest.approx(300.0, rel=1e-4)

    def test_zero_diesel(self):
        """No diesel -> Scope 1 = 0."""
        result = scope1_scope2_emissions(0, 500_000, 0.6)
        assert result["scope1_tco2"] == pytest.approx(0.0)
        assert result["scope2_tco2"] == pytest.approx(300.0, rel=1e-4)

    def test_total_is_sum(self):
        """Total should always equal scope1 + scope2."""
        result = scope1_scope2_emissions(123.4, 987_654, 0.75)
        assert result["total_tco2"] == pytest.approx(
            result["scope1_tco2"] + result["scope2_tco2"], rel=1e-6
        )

    def test_invalid_diesel(self):
        """Negative diesel raises ValueError."""
        with pytest.raises(ValueError, match="diesel_t_co2"):
            scope1_scope2_emissions(-10, 1000, 0.5)

    def test_invalid_electricity(self):
        """Negative electricity raises ValueError."""
        with pytest.raises(ValueError, match="electricity_kwh"):
            scope1_scope2_emissions(100, -1000, 0.5)
