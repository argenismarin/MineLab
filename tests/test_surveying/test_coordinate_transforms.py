"""Tests for minelab.surveying.coordinate_transforms."""

import math

import pytest

from minelab.surveying.coordinate_transforms import (
    bearing_distance,
    collar_to_downhole,
    grid_to_mine_coordinates,
    latlon_to_utm,
    utm_to_latlon,
)


class TestUtmToLatlon:
    """Tests for utm_to_latlon."""

    def test_equator_central_meridian(self):
        """500000 E, 0 N in zone 31N should be near (0, 3)."""
        result = utm_to_latlon(500000, 0, 31, "N")
        assert result["latitude"] == pytest.approx(0.0, abs=0.01)
        assert result["longitude"] == pytest.approx(3.0, abs=0.01)

    def test_roundtrip_northern(self):
        """Forward then inverse should recover original lat/lon."""
        fwd = latlon_to_utm(48.8566, 2.3522)
        inv = utm_to_latlon(
            fwd["easting"], fwd["northing"],
            fwd["zone_number"], fwd["zone_letter"],
        )
        assert inv["latitude"] == pytest.approx(48.8566, abs=0.001)
        assert inv["longitude"] == pytest.approx(2.3522, abs=0.001)

    def test_roundtrip_southern(self):
        """Southern hemisphere round-trip should also work."""
        fwd = latlon_to_utm(-33.8688, 151.2093)
        inv = utm_to_latlon(
            fwd["easting"], fwd["northing"],
            fwd["zone_number"], fwd["zone_letter"],
        )
        assert inv["latitude"] == pytest.approx(-33.8688, abs=0.001)
        assert inv["longitude"] == pytest.approx(151.2093, abs=0.001)

    def test_invalid_zone_number(self):
        """Zone number outside 1-60 should raise ValueError."""
        with pytest.raises(ValueError, match="zone_number"):
            utm_to_latlon(500000, 5000000, 0, "N")

    def test_invalid_zone_letter(self):
        """Invalid zone letter should raise ValueError."""
        with pytest.raises(ValueError, match="zone_letter"):
            utm_to_latlon(500000, 5000000, 31, "A")


class TestLatlonToUtm:
    """Tests for latlon_to_utm."""

    def test_equator_central_meridian(self):
        """(0, 3) should be 500000 E in zone 31."""
        result = latlon_to_utm(0.0, 3.0)
        assert result["easting"] == pytest.approx(500000.0, abs=1.0)
        assert result["zone_number"] == 31

    def test_known_coordinates(self):
        """Paris (48.8566, 2.3522) should be in zone 31."""
        result = latlon_to_utm(48.8566, 2.3522)
        assert result["zone_number"] == 31
        assert result["zone_letter"] >= "N"  # northern hemisphere

    def test_southern_hemisphere(self):
        """Southern hemisphere should have large northing with offset."""
        result = latlon_to_utm(-33.8688, 151.2093)
        assert result["northing"] > 6_000_000  # southern offset
        assert result["zone_letter"] < "N"

    def test_invalid_latitude(self):
        """Latitude outside [-84, 84] should raise ValueError."""
        with pytest.raises(ValueError, match="latitude"):
            latlon_to_utm(85, 0)

    def test_invalid_longitude(self):
        """Longitude outside [-180, 180] should raise ValueError."""
        with pytest.raises(ValueError, match="longitude"):
            latlon_to_utm(0, 181)


class TestGridToMineCoordinates:
    """Tests for grid_to_mine_coordinates."""

    def test_no_rotation(self):
        """Zero rotation: mine coords = grid offset."""
        result = grid_to_mine_coordinates(1100, 2050, 1000, 2000, 0)
        assert result["mine_easting"] == pytest.approx(100.0)
        assert result["mine_northing"] == pytest.approx(50.0)

    def test_90_degree_rotation(self):
        """90 deg rotation should swap axes."""
        result = grid_to_mine_coordinates(1100, 2000, 1000, 2000, 90)
        assert result["mine_easting"] == pytest.approx(0.0, abs=1e-6)
        assert result["mine_northing"] == pytest.approx(-100.0, abs=1e-6)

    def test_origin_coincident(self):
        """Point at origin gives (0, 0) regardless of rotation."""
        result = grid_to_mine_coordinates(500, 600, 500, 600, 45)
        assert result["mine_easting"] == pytest.approx(0.0, abs=1e-10)
        assert result["mine_northing"] == pytest.approx(0.0, abs=1e-10)

    def test_45_degree_rotation(self):
        """45 deg rotation of (1, 0) offset."""
        r = grid_to_mine_coordinates(1001, 2000, 1000, 2000, 45)
        c45 = math.cos(math.radians(45))
        s45 = math.sin(math.radians(45))
        assert r["mine_easting"] == pytest.approx(c45, abs=1e-4)
        assert r["mine_northing"] == pytest.approx(-s45, abs=1e-4)


class TestCollarToDownhole:
    """Tests for collar_to_downhole."""

    def test_vertical_hole(self):
        """Vertical hole (-90 dip) should only move in Z."""
        result = collar_to_downhole(1000, 2000, 500, 0, -90, [0, 50, 100])
        assert result["x"][-1] == pytest.approx(1000, abs=1e-6)
        assert result["y"][-1] == pytest.approx(2000, abs=1e-6)
        assert result["z"][-1] == pytest.approx(600, abs=1e-6)

    def test_horizontal_hole_north(self):
        """Horizontal hole (dip=0, az=0) should go due north."""
        result = collar_to_downhole(0, 0, 0, 0, 0, [100])
        assert result["x"][0] == pytest.approx(0, abs=1e-6)
        assert result["y"][0] == pytest.approx(100, abs=1e-6)
        assert result["z"][0] == pytest.approx(0, abs=1e-6)

    def test_total_depth(self):
        """Total depth should be the max interval."""
        result = collar_to_downhole(0, 0, 0, 0, -45, [0, 25, 50, 100])
        assert result["total_depth"] == pytest.approx(100.0)

    def test_invalid_dip(self):
        """Dip outside [-90, 90] should raise ValueError."""
        with pytest.raises(ValueError, match="dip"):
            collar_to_downhole(0, 0, 0, 0, -91, [10])

    def test_empty_intervals(self):
        """Empty depth intervals should raise ValueError."""
        with pytest.raises(ValueError, match="depth_intervals"):
            collar_to_downhole(0, 0, 0, 0, -90, [])


class TestBearingDistance:
    """Tests for bearing_distance."""

    def test_north(self):
        """Due north should be bearing 0."""
        result = bearing_distance(0, 0, 0, 100)
        assert result["bearing_deg"] == pytest.approx(0.0)
        assert result["distance_m"] == pytest.approx(100.0)

    def test_east(self):
        """Due east should be bearing 90."""
        result = bearing_distance(0, 0, 100, 0)
        assert result["bearing_deg"] == pytest.approx(90.0)
        assert result["distance_m"] == pytest.approx(100.0)

    def test_northeast_45(self):
        """Equal dx and dy should be bearing 45."""
        result = bearing_distance(0, 0, 100, 100)
        assert result["bearing_deg"] == pytest.approx(45.0)
        assert result["distance_m"] == pytest.approx(
            math.sqrt(20000), rel=1e-4
        )

    def test_same_point(self):
        """Same point should give zero distance."""
        result = bearing_distance(50, 50, 50, 50)
        assert result["distance_m"] == pytest.approx(0.0, abs=1e-10)

    def test_southwest(self):
        """Southwest bearing should be in (180, 270)."""
        result = bearing_distance(100, 100, 50, 50)
        assert 180 < result["bearing_deg"] < 270
