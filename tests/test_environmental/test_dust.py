"""Tests for minelab.environmental.dust."""

import math

import pytest

from minelab.environmental.dust import (
    emission_factor_haul_roads,
    gaussian_plume,
)


class TestEmissionFactorHaulRoads:
    """Tests for AP-42 emission factor."""

    def test_positive(self):
        """EF should be positive."""
        ef = emission_factor_haul_roads(8.0, 50.0)
        assert ef > 0

    def test_higher_silt_higher_ef(self):
        """Higher silt → higher EF."""
        ef_low = emission_factor_haul_roads(4.0, 50.0)
        ef_high = emission_factor_haul_roads(12.0, 50.0)
        assert ef_high > ef_low

    def test_heavier_vehicle_higher_ef(self):
        """Heavier vehicle → higher EF."""
        ef_light = emission_factor_haul_roads(8.0, 20.0)
        ef_heavy = emission_factor_haul_roads(8.0, 100.0)
        assert ef_heavy > ef_light


class TestGaussianPlume:
    """Tests for Gaussian plume dispersion."""

    def test_centerline_max(self):
        """Centerline (y=0) should give maximum concentration."""
        c_center = gaussian_plume(100, 5, 50, 20, 10, 500, 0, 0)
        c_off = gaussian_plume(100, 5, 50, 20, 10, 500, 100, 0)
        assert c_center > c_off

    def test_positive(self):
        """Concentration should be positive."""
        c = gaussian_plume(100, 5, 50, 20, 10, 500, 0, 0)
        assert c > 0

    def test_known_formula(self):
        """Manual calculation at centerline, ground level."""
        Q, u, sy, sz, H = 100, 5, 50, 20, 10
        expected = (Q / (2 * math.pi * u * sy * sz)) * (
            math.exp(-H**2 / (2 * sz**2)) + math.exp(-H**2 / (2 * sz**2))
        )
        c = gaussian_plume(Q, u, sy, sz, H, 500, 0, 0)
        assert c == pytest.approx(expected, rel=0.01)
