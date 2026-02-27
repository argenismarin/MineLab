"""Tests for minelab.utilities.conversions."""

import math

import numpy as np
import pytest

from minelab.utilities.conversions import (
    angle_convert,
    density_convert,
    energy_convert,
    flowrate_convert,
    length_convert,
    mass_convert,
    pressure_convert,
    temperature_convert,
    volume_convert,
)


# ---- Length ----------------------------------------------------------------

class TestLengthConvert:
    def test_identity(self):
        assert length_convert(5.0, "m", "m") == 5.0

    def test_ft_to_m(self):
        assert pytest.approx(length_convert(1, "ft", "m"), abs=1e-6) == 0.3048

    def test_m_to_ft(self):
        assert pytest.approx(length_convert(1, "m", "ft"), rel=1e-6) == 1 / 0.3048

    def test_yd_to_ft(self):
        assert pytest.approx(length_convert(1, "yd", "ft"), rel=1e-9) == 3.0

    def test_in_to_cm(self):
        assert pytest.approx(length_convert(1, "in", "cm"), rel=1e-9) == 2.54

    def test_round_trip(self):
        val = 123.456
        converted = length_convert(length_convert(val, "m", "ft"), "ft", "m")
        assert pytest.approx(converted, rel=1e-12) == val

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = length_convert(arr, "m", "cm")
        np.testing.assert_allclose(result, [100, 200, 300])

    def test_unknown_unit(self):
        with pytest.raises(ValueError, match="Unknown length unit"):
            length_convert(1, "m", "furlongs")


# ---- Mass ------------------------------------------------------------------

class TestMassConvert:
    def test_tonne_to_kg(self):
        assert mass_convert(1, "tonne", "kg") == 1000.0

    def test_lb_to_kg(self):
        assert pytest.approx(mass_convert(1, "lb", "kg"), abs=1e-6) == 0.453592

    def test_ton_to_lb(self):
        assert pytest.approx(mass_convert(1, "ton", "lb"), rel=1e-9) == 2000.0

    def test_round_trip_oz(self):
        val = 42.0
        converted = mass_convert(mass_convert(val, "oz", "g"), "g", "oz")
        assert pytest.approx(converted, rel=1e-12) == val


# ---- Volume ----------------------------------------------------------------

class TestVolumeConvert:
    def test_m3_to_L(self):
        assert volume_convert(1, "m3", "L") == 1000.0

    def test_gal_to_L(self):
        assert pytest.approx(volume_convert(1, "gal", "L"), rel=1e-6) == 3.785411784

    def test_bbl_to_gal(self):
        assert pytest.approx(volume_convert(1, "bbl", "gal"), rel=1e-4) == 42.0

    def test_round_trip(self):
        val = 7.5
        converted = volume_convert(volume_convert(val, "ft3", "L"), "L", "ft3")
        assert pytest.approx(converted, rel=1e-12) == val


# ---- Pressure --------------------------------------------------------------

class TestPressureConvert:
    def test_atm_to_Pa(self):
        assert pressure_convert(1, "atm", "Pa") == 101325.0

    def test_atm_to_bar(self):
        assert pytest.approx(pressure_convert(1, "atm", "bar"), rel=1e-6) == 1.01325

    def test_psi_to_kPa(self):
        assert pytest.approx(pressure_convert(1, "psi", "kPa"), rel=1e-4) == 6.89476

    def test_MPa_to_psi(self):
        val = 10.0
        converted = pressure_convert(pressure_convert(val, "MPa", "psi"), "psi", "MPa")
        assert pytest.approx(converted, rel=1e-12) == val


# ---- Density ---------------------------------------------------------------

class TestDensityConvert:
    def test_gcm3_to_kgm3(self):
        assert density_convert(1, "g/cm3", "kg/m3") == 1000.0

    def test_tm3_to_gcm3(self):
        assert pytest.approx(density_convert(2.65, "t/m3", "g/cm3"), rel=1e-9) == 2.65

    def test_round_trip(self):
        val = 160.0
        converted = density_convert(
            density_convert(val, "lb/ft3", "kg/m3"), "kg/m3", "lb/ft3"
        )
        assert pytest.approx(converted, rel=1e-9) == val


# ---- Angle -----------------------------------------------------------------

class TestAngleConvert:
    def test_deg_to_rad(self):
        assert pytest.approx(angle_convert(180, "deg", "rad"), rel=1e-9) == math.pi

    def test_rad_to_deg(self):
        assert pytest.approx(angle_convert(math.pi, "rad", "deg"), rel=1e-9) == 180.0

    def test_grad_to_deg(self):
        assert pytest.approx(angle_convert(100, "grad", "deg"), rel=1e-9) == 90.0

    def test_deg_to_grad(self):
        assert pytest.approx(angle_convert(90, "deg", "grad"), rel=1e-9) == 100.0


# ---- Energy ----------------------------------------------------------------

class TestEnergyConvert:
    def test_kWh_to_J(self):
        assert energy_convert(1, "kWh", "J") == 3_600_000.0

    def test_BTU_to_J(self):
        assert pytest.approx(energy_convert(1, "BTU", "J"), rel=1e-4) == 1055.056

    def test_round_trip(self):
        val = 500.0
        converted = energy_convert(energy_convert(val, "cal", "kJ"), "kJ", "cal")
        assert pytest.approx(converted, rel=1e-12) == val


# ---- Flowrate --------------------------------------------------------------

class TestFlowrateConvert:
    def test_m3s_to_m3h(self):
        assert flowrate_convert(1, "m3/s", "m3/h") == 3600.0

    def test_gpm_to_Lmin(self):
        assert pytest.approx(
            flowrate_convert(1, "gpm", "L/min"), rel=1e-4
        ) == 3.78541

    def test_round_trip(self):
        val = 100.0
        converted = flowrate_convert(
            flowrate_convert(val, "cfm", "m3/s"), "m3/s", "cfm"
        )
        assert pytest.approx(converted, rel=1e-9) == val


# ---- Temperature -----------------------------------------------------------

class TestTemperatureConvert:
    def test_C_to_F_freezing(self):
        assert temperature_convert(0, "C", "F") == 32.0

    def test_C_to_F_boiling(self):
        assert temperature_convert(100, "C", "F") == 212.0

    def test_F_to_C(self):
        assert temperature_convert(32, "F", "C") == 0.0

    def test_C_to_K(self):
        assert temperature_convert(0, "C", "K") == 273.15

    def test_K_to_C(self):
        assert temperature_convert(273.15, "K", "C") == 0.0

    def test_F_to_K(self):
        assert pytest.approx(temperature_convert(32, "F", "K"), rel=1e-9) == 273.15

    def test_identity(self):
        assert temperature_convert(37.0, "C", "C") == 37.0

    def test_round_trip(self):
        val = -40.0  # famous crossover point
        assert pytest.approx(
            temperature_convert(val, "C", "F"), rel=1e-9
        ) == -40.0

    def test_unknown_unit(self):
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            temperature_convert(100, "C", "R")

    def test_numpy_array(self):
        arr = np.array([0, 100])
        result = temperature_convert(arr, "C", "F")
        np.testing.assert_allclose(result, [32, 212])
