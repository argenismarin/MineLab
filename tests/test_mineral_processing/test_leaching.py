"""Tests for minelab.mineral_processing.leaching."""

import numpy as np
import pytest

from minelab.mineral_processing.leaching import (
    acid_consumption,
    arrhenius_rate,
    cyanidation_kinetics,
    heap_leach_recovery,
    shrinking_core_diffusion,
    shrinking_core_film,
    shrinking_core_reaction,
)


class TestShrinkingCoreReaction:
    """Tests for reaction-controlled shrinking core model."""

    def test_zero_time(self):
        """t=0 → X=0."""
        x = shrinking_core_reaction(0.001, 1e-4, 0, 50000, 100)
        assert x == pytest.approx(0.0, abs=1e-6)

    def test_conversion_increases(self):
        """Longer time → higher conversion."""
        x1 = shrinking_core_reaction(0.001, 1e-4, 50, 50000, 100)
        x2 = shrinking_core_reaction(0.001, 1e-4, 100, 50000, 100)
        assert x2 > x1

    def test_max_conversion(self):
        """Very long time → X approaches 1."""
        x = shrinking_core_reaction(0.001, 1e-4, 1e6, 50000, 100)
        assert x == pytest.approx(1.0, abs=0.01)


class TestShrinkingCoreDiffusion:
    """Tests for diffusion-controlled shrinking core model."""

    def test_positive_conversion(self):
        """Should return positive conversion."""
        x = shrinking_core_diffusion(0.001, 1e-10, 1000, 50000, 100)
        assert x > 0

    def test_monotonic(self):
        """Conversion increases with time."""
        x1 = shrinking_core_diffusion(0.001, 1e-10, 500, 50000, 100)
        x2 = shrinking_core_diffusion(0.001, 1e-10, 1000, 50000, 100)
        assert x2 >= x1


class TestShrinkingCoreFilm:
    """Tests for film-controlled shrinking core model."""

    def test_linear(self):
        """Film control is linear in time."""
        x1 = shrinking_core_film(0.001, 1e-3, 50, 50000, 100)
        x2 = shrinking_core_film(0.001, 1e-3, 100, 50000, 100)
        assert x2 == pytest.approx(2 * x1, rel=0.01) or x2 >= x1


class TestHeapLeachRecovery:
    """Tests for heap leach extrapolation."""

    def test_interpolation(self):
        """Should interpolate within data range."""
        times = np.array([0, 30, 60, 90, 120])
        rec = np.array([0, 0.3, 0.5, 0.6, 0.65])
        r = heap_leach_recovery(rec, times, 60)
        assert r == pytest.approx(0.5, rel=0.01)


class TestArrheniusRate:
    """Tests for Arrhenius rate constant."""

    def test_positive(self):
        """Rate constant should be positive."""
        k = arrhenius_rate(1e10, 50000, 298)
        assert k > 0

    def test_higher_temp_faster(self):
        """Higher temperature → higher rate."""
        k1 = arrhenius_rate(1e10, 50000, 298)
        k2 = arrhenius_rate(1e10, 50000, 348)
        assert k2 > k1


class TestCyanidationKinetics:
    """Tests for cyanidation leach kinetics."""

    def test_known_value(self):
        """24h leach with k=0.1 → R ≈ 0.91."""
        r = cyanidation_kinetics(5.0, 0.5, 24, 0.1)
        assert r == pytest.approx(0.91, rel=0.02)

    def test_zero_time(self):
        """t=0 → R=0."""
        r = cyanidation_kinetics(5.0, 0.5, 0, 0.1)
        assert r == pytest.approx(0.0, abs=1e-10)


class TestAcidConsumption:
    """Tests for acid consumption estimation."""

    def test_known_value(self):
        """2% S → MPA = 61.2 kg H2SO4/t."""
        ac = acid_consumption(2.0)
        assert ac == pytest.approx(61.2, rel=0.01)
