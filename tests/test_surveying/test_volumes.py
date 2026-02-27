"""Tests for minelab.surveying.volumes."""

import math

import pytest

from minelab.surveying.volumes import (
    cone_stockpile_volume,
    end_area_volume,
    prismatoid_volume,
    stockpile_mass,
    trapezoidal_cross_section_area,
)


class TestPrismatoidVolume:
    """Tests for prismatoid_volume."""

    def test_known_value(self):
        """V = 10/6 * (100 + 4*80 + 60) = 800 m3."""
        result = prismatoid_volume(100, 80, 60, 10)
        expected = 10 / 6.0 * (100 + 4 * 80 + 60)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_uniform_areas(self):
        """Equal areas should give V = A * h."""
        result = prismatoid_volume(50, 50, 50, 20)
        assert result == pytest.approx(50 * 20, rel=1e-4)

    def test_zero_top_area(self):
        """Zero top area (pyramid-like) should still be valid."""
        result = prismatoid_volume(0, 25, 100, 6)
        assert result > 0

    def test_invalid_height(self):
        """Non-positive height should raise ValueError."""
        with pytest.raises(ValueError, match="height"):
            prismatoid_volume(100, 80, 60, 0)

    def test_negative_area(self):
        """Negative area should raise ValueError."""
        with pytest.raises(ValueError, match="a_top"):
            prismatoid_volume(-10, 80, 60, 10)


class TestConeStockpileVolume:
    """Tests for cone_stockpile_volume."""

    def test_known_value(self):
        """V = pi/3 * 10^2 * 5 = 523.599 m3."""
        result = cone_stockpile_volume(10, 5, 30)
        expected = math.pi / 3.0 * 100 * 5
        assert result["volume_m3"] == pytest.approx(expected, rel=1e-4)

    def test_surface_area(self):
        """Surface area = pi * r * sqrt(r^2 + h^2)."""
        result = cone_stockpile_volume(10, 5, 30)
        slant = math.sqrt(100 + 25)
        expected = math.pi * 10 * slant
        assert result["surface_area_m2"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_base_area(self):
        """Base area = pi * r^2."""
        result = cone_stockpile_volume(10, 5, 30)
        assert result["base_area_m2"] == pytest.approx(
            math.pi * 100, rel=1e-4
        )

    def test_invalid_radius(self):
        """Non-positive radius should raise ValueError."""
        with pytest.raises(ValueError, match="base_radius"):
            cone_stockpile_volume(0, 5, 30)

    def test_invalid_angle_of_repose(self):
        """Angle >= 90 should raise ValueError."""
        with pytest.raises(ValueError, match="angle_of_repose"):
            cone_stockpile_volume(10, 5, 90)


class TestTrapezoidalCrossSectionArea:
    """Tests for trapezoidal_cross_section_area."""

    def test_known_value(self):
        """(4 + 8)/2 * 3 = 18 m2."""
        result = trapezoidal_cross_section_area(4, 8, 3)
        assert result == pytest.approx(18.0)

    def test_rectangle(self):
        """Equal widths should give rectangular area."""
        result = trapezoidal_cross_section_area(5, 5, 4)
        assert result == pytest.approx(20.0)

    def test_triangle(self):
        """Zero bottom width gives triangular cross section."""
        result = trapezoidal_cross_section_area(0, 6, 4)
        assert result == pytest.approx(12.0)

    def test_invalid_height(self):
        """Non-positive height should raise ValueError."""
        with pytest.raises(ValueError, match="height"):
            trapezoidal_cross_section_area(4, 8, 0)


class TestEndAreaVolume:
    """Tests for end_area_volume."""

    def test_known_value(self):
        """V = (50+70)/2*10 + (70+60)/2*15 = 600 + 975 = 1575 m3."""
        result = end_area_volume([50, 70, 60], [10, 15])
        assert result == pytest.approx(1575.0)

    def test_uniform_areas_and_distances(self):
        """Equal areas and distances: V = A * total_distance."""
        result = end_area_volume([100, 100, 100], [20, 20])
        assert result == pytest.approx(4000.0)

    def test_two_sections(self):
        """Simplest case: two sections, one distance."""
        result = end_area_volume([40, 60], [10])
        assert result == pytest.approx(500.0)

    def test_mismatched_lengths(self):
        """Mismatched areas and distances should raise ValueError."""
        with pytest.raises(ValueError, match="distances"):
            end_area_volume([50, 70, 60], [10])

    def test_single_area(self):
        """Single area element should raise ValueError."""
        with pytest.raises(ValueError, match="areas"):
            end_area_volume([50], [])


class TestStockpileMass:
    """Tests for stockpile_mass."""

    def test_known_value(self):
        """1000 m3 * 1.8 t/m3 = 1800 t dry."""
        result = stockpile_mass(1000, 1.8, 5)
        assert result["dry_mass_tonnes"] == pytest.approx(1800.0)
        assert result["wet_mass_tonnes"] == pytest.approx(1890.0)
        assert result["moisture_tonnes"] == pytest.approx(90.0)

    def test_zero_moisture(self):
        """Zero moisture: wet = dry mass."""
        result = stockpile_mass(500, 2.0, 0)
        assert result["dry_mass_tonnes"] == pytest.approx(1000.0)
        assert result["wet_mass_tonnes"] == pytest.approx(1000.0)
        assert result["moisture_tonnes"] == pytest.approx(0.0)

    def test_moisture_positive(self):
        """Moisture content should always be non-negative."""
        result = stockpile_mass(100, 1.5, 10)
        assert result["moisture_tonnes"] >= 0

    def test_invalid_volume(self):
        """Non-positive volume should raise ValueError."""
        with pytest.raises(ValueError, match="volume"):
            stockpile_mass(0, 1.8, 5)

    def test_invalid_density(self):
        """Non-positive density should raise ValueError."""
        with pytest.raises(ValueError, match="density"):
            stockpile_mass(1000, -1.0, 5)
