"""Tests for minelab.ventilation.airway_resistance."""

import pytest

from minelab.ventilation.airway_resistance import (
    atkinson_resistance,
    friction_factor_from_roughness,
    natural_ventilation_pressure,
    parallel_resistance,
    pressure_drop,
    series_resistance,
)


class TestAtkinsonResistance:
    """Tests for Atkinson equation."""

    def test_known_value(self):
        """k=0.012, L=500, Per=12, A=9 → R ≈ 0.0988."""
        r = atkinson_resistance(0.012, 500, 12, 9)
        # R = 0.012 * 500 * 12 / 9^3 = 72 / 729 ≈ 0.0988
        assert r == pytest.approx(0.0988, rel=0.01)

    def test_positive(self):
        """Resistance should be positive."""
        r = atkinson_resistance(0.01, 100, 10, 8)
        assert r > 0


class TestPressureDrop:
    """Tests for pressure drop."""

    def test_known_value(self):
        """R=0.5, Q=50 → ΔP = 1250 Pa."""
        dp = pressure_drop(0.5, 50)
        assert dp == pytest.approx(1250, rel=0.01)

    def test_zero_flow(self):
        """Q=0 → ΔP=0."""
        dp = pressure_drop(0.5, 0)
        assert dp == pytest.approx(0.0)


class TestSeriesResistance:
    """Tests for series resistance."""

    def test_known(self):
        """[1, 2, 3] → 6."""
        r = series_resistance([1, 2, 3])
        assert r == pytest.approx(6.0)

    def test_single(self):
        """Single → itself."""
        r = series_resistance([5.0])
        assert r == pytest.approx(5.0)


class TestParallelResistance:
    """Tests for parallel resistance."""

    def test_equal(self):
        """[4, 4] → 1.0."""
        r = parallel_resistance([4, 4])
        assert r == pytest.approx(1.0, rel=0.01)

    def test_single(self):
        """Single → itself."""
        r = parallel_resistance([5.0])
        assert r == pytest.approx(5.0)


class TestFrictionFactor:
    """Tests for friction factor estimation."""

    def test_positive(self):
        """Should return positive value."""
        k = friction_factor_from_roughness(0.05, 3.0)
        assert k > 0


class TestNVP:
    """Tests for natural ventilation pressure."""

    def test_positive_nvp(self):
        """Hot underground → positive NVP (upcast)."""
        nvp = natural_ventilation_pressure([500], [15.0], [30.0])
        assert nvp > 0
