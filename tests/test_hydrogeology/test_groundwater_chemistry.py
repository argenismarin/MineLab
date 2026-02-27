"""Tests for minelab.hydrogeology.groundwater_chemistry module."""


import pytest

from minelab.hydrogeology.groundwater_chemistry import (
    acid_mine_drainage_rate,
    dilution_attenuation_factor,
    langelier_index,
    mass_balance_water_quality,
    seepage_velocity,
)

# -------------------------------------------------------------------------
# Acid Mine Drainage Rate
# -------------------------------------------------------------------------


class TestAcidMineDrainageRate:
    """Tests for the Nicholson et al. (1988) AMD rate model."""

    def test_known_value(self):
        """AMD = pyrite_frac * SA * ox_rate * density.

        0.05 * 10.0 * 0.001 * 2500 = 1.25 kg/day per m3.
        """
        result = acid_mine_drainage_rate(
            pyrite_fraction=0.05,
            surface_area=10.0,
            oxidation_rate=0.001,
            density=2500.0,
        )
        assert result == pytest.approx(1.25)

    def test_zero_pyrite(self):
        """No pyrite should produce zero AMD."""
        result = acid_mine_drainage_rate(0.0, 10.0, 0.001, 2500.0)
        assert result == pytest.approx(0.0)

    def test_higher_pyrite_increases_rate(self):
        """More pyrite should produce higher AMD rate."""
        r1 = acid_mine_drainage_rate(0.02, 10.0, 0.001, 2500.0)
        r2 = acid_mine_drainage_rate(0.10, 10.0, 0.001, 2500.0)
        assert r2 > r1

    def test_invalid_pyrite_fraction(self):
        with pytest.raises(ValueError, match="pyrite_fraction"):
            acid_mine_drainage_rate(-0.01, 10.0, 0.001, 2500.0)

    def test_invalid_pyrite_fraction_over_one(self):
        with pytest.raises(ValueError, match="pyrite_fraction"):
            acid_mine_drainage_rate(1.5, 10.0, 0.001, 2500.0)

    def test_invalid_surface_area(self):
        with pytest.raises(ValueError, match="surface_area"):
            acid_mine_drainage_rate(0.05, 0.0, 0.001, 2500.0)


# -------------------------------------------------------------------------
# Dilution Attenuation Factor
# -------------------------------------------------------------------------


class TestDilutionAttenuationFactor:
    """Tests for the retardation-based DAF."""

    def test_known_value(self):
        """R = 1 + rho_bulk * K_d / theta.

        R = 1 + 1.6 * 5.0 / 0.3 = 1 + 26.667 = 27.667.
        """
        result = dilution_attenuation_factor(
            C_source=100.0, K_d=5.0, rho_bulk=1.6, theta=0.3
        )
        assert result == pytest.approx(27.667, rel=1e-3)

    def test_no_sorption(self):
        """With K_d=0, retardation factor should be 1."""
        result = dilution_attenuation_factor(
            C_source=100.0, K_d=0.0, rho_bulk=1.6, theta=0.3
        )
        assert result == pytest.approx(1.0)

    def test_higher_sorption_increases_daf(self):
        """Greater sorption should increase attenuation."""
        daf1 = dilution_attenuation_factor(100.0, 1.0, 1.6, 0.3)
        daf2 = dilution_attenuation_factor(100.0, 10.0, 1.6, 0.3)
        assert daf2 > daf1

    def test_invalid_source_concentration(self):
        with pytest.raises(ValueError, match="C_source"):
            dilution_attenuation_factor(0.0, 5.0, 1.6, 0.3)

    def test_invalid_theta(self):
        with pytest.raises(ValueError, match="theta"):
            dilution_attenuation_factor(100.0, 5.0, 1.6, 0.0)


# -------------------------------------------------------------------------
# Seepage Velocity
# -------------------------------------------------------------------------


class TestSeepageVelocity:
    """Tests for seepage (pore water) velocity."""

    def test_known_value(self):
        """v = K * i / n = 5.0 * 0.01 / 0.25 = 0.2 m/day."""
        result = seepage_velocity(K=5.0, gradient=0.01, porosity=0.25)
        assert result == pytest.approx(0.2)

    def test_higher_gradient_increases_velocity(self):
        """Steeper gradient should increase velocity."""
        v1 = seepage_velocity(K=5.0, gradient=0.01, porosity=0.25)
        v2 = seepage_velocity(K=5.0, gradient=0.05, porosity=0.25)
        assert v2 > v1

    def test_higher_porosity_decreases_velocity(self):
        """Greater porosity should decrease velocity for same Darcy flux."""
        v1 = seepage_velocity(K=5.0, gradient=0.01, porosity=0.1)
        v2 = seepage_velocity(K=5.0, gradient=0.01, porosity=0.5)
        assert v1 > v2

    def test_invalid_conductivity(self):
        with pytest.raises(ValueError, match="K"):
            seepage_velocity(K=-1.0, gradient=0.01, porosity=0.25)

    def test_invalid_porosity(self):
        with pytest.raises(ValueError, match="porosity"):
            seepage_velocity(K=5.0, gradient=0.01, porosity=0.0)


# -------------------------------------------------------------------------
# Langelier Saturation Index
# -------------------------------------------------------------------------


class TestLangelierIndex:
    """Tests for the Langelier Saturation Index."""

    def test_known_value(self):
        """Typical municipal water: pH=7.5, 25C, Ca=200, Alk=100, TDS=500.

        A = (log10(500) - 1) / 10 = (2.6990 - 1) / 10 = 0.1699
        B = -13.12 * log10(25+273) + 34.55
          = -13.12 * log10(298) + 34.55
          = -13.12 * 2.4742 + 34.55
          = -32.462 + 34.55 = 2.088
        C = log10(200) - 0.4 = 2.3010 - 0.4 = 1.901
        D = log10(100) = 2.0
        pHs = (9.3 + 0.1699 + 2.088) - (1.901 + 2.0) = 11.558 - 3.901
            = 7.657
        LSI = 7.5 - 7.657 = -0.157.
        """
        result = langelier_index(
            pH=7.5, temp_c=25.0, ca_ppm=200.0,
            total_alk_ppm=100.0, tds_ppm=500.0,
        )
        assert result == pytest.approx(-0.157, abs=0.02)

    def test_scale_forming_water(self):
        """High pH, high Ca, high Alk -> positive LSI."""
        result = langelier_index(
            pH=8.5, temp_c=40.0, ca_ppm=400.0,
            total_alk_ppm=300.0, tds_ppm=1000.0,
        )
        assert result > 0

    def test_corrosive_water(self):
        """Low pH, low Ca, low Alk -> negative LSI."""
        result = langelier_index(
            pH=6.0, temp_c=10.0, ca_ppm=20.0,
            total_alk_ppm=20.0, tds_ppm=100.0,
        )
        assert result < 0

    def test_invalid_ph_value(self):
        with pytest.raises(ValueError, match="pH"):
            langelier_index(
                pH=15.0, temp_c=25.0, ca_ppm=200.0,
                total_alk_ppm=100.0, tds_ppm=500.0,
            )

    def test_invalid_temp(self):
        with pytest.raises(ValueError, match="temp_c"):
            langelier_index(
                pH=7.5, temp_c=0.0, ca_ppm=200.0,
                total_alk_ppm=100.0, tds_ppm=500.0,
            )


# -------------------------------------------------------------------------
# Mass Balance Water Quality
# -------------------------------------------------------------------------


class TestMassBalanceWaterQuality:
    """Tests for flow-weighted mixing calculation."""

    def test_known_value(self):
        """Two streams: Q=[100, 200], C=[50, 20].

        total_flow = 300
        C_mix = (100*50 + 200*20) / 300 = (5000+4000)/300 = 30.0.
        """
        result = mass_balance_water_quality(
            flows=[100.0, 200.0], concentrations=[50.0, 20.0]
        )
        assert result["total_flow"] == pytest.approx(300.0)
        assert result["mixed_concentration"] == pytest.approx(30.0)

    def test_single_stream(self):
        """Single stream should return its own concentration."""
        result = mass_balance_water_quality(
            flows=[500.0], concentrations=[75.0]
        )
        assert result["total_flow"] == pytest.approx(500.0)
        assert result["mixed_concentration"] == pytest.approx(75.0)

    def test_equal_flows(self):
        """Equal flows => simple arithmetic mean of concentrations."""
        result = mass_balance_water_quality(
            flows=[100.0, 100.0, 100.0],
            concentrations=[10.0, 20.0, 30.0],
        )
        assert result["mixed_concentration"] == pytest.approx(20.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            mass_balance_water_quality(
                flows=[100.0, 200.0], concentrations=[50.0]
            )

    def test_negative_flow_raises(self):
        with pytest.raises(ValueError, match="positive"):
            mass_balance_water_quality(
                flows=[-100.0, 200.0], concentrations=[50.0, 20.0]
            )

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mass_balance_water_quality(
                flows=[100.0, 200.0], concentrations=[-5.0, 20.0]
            )

    def test_empty_flows_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            mass_balance_water_quality(flows=[], concentrations=[])
