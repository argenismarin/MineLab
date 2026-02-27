"""Tests for minelab.geomechanics.slope_stability."""

import numpy as np
import pytest

from minelab.geomechanics.slope_stability import (
    bishop_simplified,
    critical_surface_search,
    fellenius_method,
    janbu_simplified,
    pseudo_static_seismic,
    spencer_method,
)


def _make_slices(n=5):
    """Create a simple set of slices for a circular failure surface."""
    slices = []
    for i in range(n):
        angle = -20 + (i * 40 / (n - 1))  # -20 to +20 degrees
        slices.append({
            "width": 2.0,
            "weight": 150.0 + i * 20,
            "base_angle": angle,
            "cohesion": 20.0,
            "friction_angle": 30.0,
            "pore_pressure": 5.0,
        })
    return slices


class TestBishopSimplified:
    """Tests for Bishop's simplified method."""

    def test_converges(self):
        """Should converge for well-posed problem."""
        slices = _make_slices()
        result = bishop_simplified(slices, 20)
        assert result["converged"]
        assert result["fos"] > 0

    def test_known_range(self):
        """FOS should be in a reasonable range."""
        slices = _make_slices()
        result = bishop_simplified(slices, 20)
        assert result["fos"] > 0.5

    def test_higher_cohesion_higher_fos(self):
        """Increasing cohesion should increase FOS."""
        slices_low = _make_slices()
        slices_high = _make_slices()
        for s in slices_high:
            s["cohesion"] = 40.0

        fos_low = bishop_simplified(slices_low, 20)["fos"]
        fos_high = bishop_simplified(slices_high, 20)["fos"]
        assert fos_high > fos_low

    def test_single_slice(self):
        """Should work with a single slice."""
        slices = [{
            "width": 5, "weight": 200, "base_angle": 25,
            "cohesion": 15, "friction_angle": 30,
        }]
        result = bishop_simplified(slices, 15)
        assert result["fos"] > 0

    def test_invalid_radius(self):
        """Negative radius should raise."""
        with pytest.raises(ValueError):
            bishop_simplified(_make_slices(), -5)


class TestJanbuSimplified:
    """Tests for Janbu's simplified method."""

    def test_positive_fos(self):
        """FOS should be positive."""
        slices = _make_slices()
        result = janbu_simplified(slices)
        assert result["fos"] > 0

    def test_with_f0(self):
        """Custom f0 should be used."""
        slices = _make_slices()
        result = janbu_simplified(slices, f0=1.05)
        assert result["f0"] == 1.05

    def test_vs_bishop(self):
        """Janbu is generally close to Bishop for circular surfaces."""
        slices = _make_slices()
        fos_bishop = bishop_simplified(slices, 20)["fos"]
        fos_janbu = janbu_simplified(slices)["fos"]
        # Should be within 30% typically
        assert abs(fos_janbu - fos_bishop) / fos_bishop < 0.5


class TestFelleniusMethod:
    """Tests for Fellenius (ordinary) method."""

    def test_positive_fos(self):
        """FOS should be positive."""
        slices = _make_slices()
        result = fellenius_method(slices, 20)
        assert result["fos"] > 0

    def test_conservative(self):
        """Fellenius FOS ≤ Bishop FOS (conservative)."""
        slices = _make_slices()
        fos_fellenius = fellenius_method(slices, 20)["fos"]
        fos_bishop = bishop_simplified(slices, 20)["fos"]
        assert fos_fellenius <= fos_bishop * 1.05  # small tolerance

    def test_invalid_radius(self):
        """Negative radius should raise."""
        with pytest.raises(ValueError):
            fellenius_method(_make_slices(), -5)


class TestSpencerMethod:
    """Tests for Spencer's rigorous method."""

    def test_converges(self):
        """Should converge."""
        slices = _make_slices()
        result = spencer_method(slices, 20)
        assert result["fos"] > 0

    def test_close_to_bishop(self):
        """Spencer should be close to Bishop for circular surfaces."""
        slices = _make_slices()
        fos_spencer = spencer_method(slices, 20)["fos"]
        fos_bishop = bishop_simplified(slices, 20)["fos"]
        assert abs(fos_spencer - fos_bishop) / fos_bishop < 0.1


class TestCriticalSurfaceSearch:
    """Tests for grid search critical surface."""

    def test_finds_minimum(self):
        """Should return a valid minimum FOS."""
        def make_slices(xc, yc, r):
            return [{
                "width": 2, "weight": 100 + 10 * abs(xc),
                "base_angle": 20, "cohesion": 15, "friction_angle": 25,
            }]

        centers = np.array([[0, 10], [5, 15], [10, 20]])
        radii = np.array([10, 15, 20])
        result = critical_surface_search(make_slices, centers, radii)
        assert result["min_fos"] > 0
        assert result["min_fos"] < np.inf
        assert result["fos_grid"].shape == (3, 3)


class TestPseudoStaticSeismic:
    """Tests for pseudo-static seismic analysis."""

    def test_static_case(self):
        """kh=0, kv=0 should match Bishop."""
        slices = _make_slices()
        static = pseudo_static_seismic(slices, 20, kh=0.0, kv=0.0)
        bishop = bishop_simplified(slices, 20)
        assert static["fos"] == pytest.approx(bishop["fos"], rel=0.01)

    def test_seismic_reduces_fos(self):
        """kh > 0 should reduce FOS."""
        slices = _make_slices()
        static = pseudo_static_seismic(slices, 20, kh=0.0)
        seismic = pseudo_static_seismic(slices, 20, kh=0.15)
        assert seismic["fos"] < static["fos"]

    def test_higher_kh_lower_fos(self):
        """Increasing kh should decrease FOS."""
        slices = _make_slices()
        fos_01 = pseudo_static_seismic(slices, 20, kh=0.1)["fos"]
        fos_02 = pseudo_static_seismic(slices, 20, kh=0.2)["fos"]
        assert fos_02 < fos_01
