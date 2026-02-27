"""Tests for minelab.surveying.blast_survey."""

import math

import pytest

from minelab.surveying.blast_survey import (
    blast_movement_vector,
    dig_rate_survey,
    drill_deviation,
    muckpile_swell_factor,
)


class TestDrillDeviation:
    """Tests for drill_deviation."""

    def test_known_value(self):
        """Deviation for collar (100,200) to toe (101,210)."""
        result = drill_deviation(100, 200, 101, 210, 0, -90)
        expected_az = math.degrees(math.atan2(1, 10))
        expected_dist = math.sqrt(1 + 100)
        assert result["actual_azimuth_deg"] == pytest.approx(
            expected_az, rel=1e-4
        )
        assert result["horizontal_displacement_m"] == pytest.approx(
            expected_dist, rel=1e-4
        )

    def test_no_deviation(self):
        """Due north should give zero azimuth deviation for az=0 design."""
        result = drill_deviation(0, 0, 0, 10, 0, -90)
        assert result["actual_azimuth_deg"] == pytest.approx(0.0, abs=1e-6)
        assert result["azimuth_deviation_deg"] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_due_east(self):
        """Movement due east should give 90 deg azimuth."""
        result = drill_deviation(0, 0, 10, 0, 0, -90)
        assert result["actual_azimuth_deg"] == pytest.approx(90.0)

    def test_coincident_points(self):
        """Coincident collar and toe should give zero displacement."""
        result = drill_deviation(100, 200, 100, 200, 0, -90)
        assert result["horizontal_displacement_m"] == pytest.approx(
            0.0, abs=1e-10
        )

    def test_deviation_sign(self):
        """Clockwise deviation should be positive."""
        result = drill_deviation(0, 0, 5, 10, 0, -90)
        assert result["azimuth_deviation_deg"] > 0


class TestBlastMovementVector:
    """Tests for blast_movement_vector."""

    def test_known_value(self):
        """Movement from (1000,2000) to (1005,2003)."""
        result = blast_movement_vector(1000, 2000, 1005, 2003)
        expected_mag = math.sqrt(25 + 9)
        assert result["magnitude_m"] == pytest.approx(
            expected_mag, rel=1e-4
        )
        assert result["dx_m"] == pytest.approx(5.0)
        assert result["dy_m"] == pytest.approx(3.0)

    def test_no_movement(self):
        """No movement should give zero magnitude."""
        result = blast_movement_vector(100, 200, 100, 200)
        assert result["magnitude_m"] == pytest.approx(0.0, abs=1e-10)

    def test_due_north_direction(self):
        """Movement due north should be 0 deg."""
        result = blast_movement_vector(0, 0, 0, 10)
        assert result["direction_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_due_east_direction(self):
        """Movement due east should be 90 deg."""
        result = blast_movement_vector(0, 0, 10, 0)
        assert result["direction_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_components(self):
        """dx and dy should be consistent with inputs."""
        result = blast_movement_vector(10, 20, 15, 30)
        assert result["dx_m"] == pytest.approx(5.0)
        assert result["dy_m"] == pytest.approx(10.0)


class TestMuckpileSwellFactor:
    """Tests for muckpile_swell_factor."""

    def test_known_value(self):
        """SF = 2.7 / 1.8 = 1.5."""
        result = muckpile_swell_factor(2.7, 1.8)
        assert result == pytest.approx(1.5, rel=1e-4)

    def test_no_swell(self):
        """Equal densities should give swell factor of 1."""
        result = muckpile_swell_factor(2.0, 2.0)
        assert result == pytest.approx(1.0)

    def test_greater_than_one(self):
        """Swell factor should be >= 1 for typical rock."""
        result = muckpile_swell_factor(2.65, 1.6)
        assert result > 1.0

    def test_invalid_in_situ(self):
        """Non-positive in-situ density should raise ValueError."""
        with pytest.raises(ValueError, match="in_situ_density"):
            muckpile_swell_factor(0, 1.8)

    def test_invalid_broken(self):
        """Non-positive broken density should raise ValueError."""
        with pytest.raises(ValueError, match="broken_density"):
            muckpile_swell_factor(2.7, 0)


class TestDigRateSurvey:
    """Tests for dig_rate_survey."""

    def test_known_value(self):
        """5000 t / 10 h = 500 t/h; 5000 / 250 = 20 t/pass."""
        result = dig_rate_survey(5000, 10, 250)
        assert result["tonnes_per_hour"] == pytest.approx(500.0)
        assert result["tonnes_per_pass"] == pytest.approx(20.0)
        assert result["passes_per_hour"] == pytest.approx(25.0)

    def test_higher_tonnage_higher_rate(self):
        """More tonnes mined should give higher rate."""
        r1 = dig_rate_survey(3000, 10, 200)
        r2 = dig_rate_survey(6000, 10, 200)
        assert r2["tonnes_per_hour"] > r1["tonnes_per_hour"]

    def test_consistency(self):
        """t/h = t/pass * passes/h."""
        result = dig_rate_survey(8000, 16, 400)
        assert result["tonnes_per_hour"] == pytest.approx(
            result["tonnes_per_pass"] * result["passes_per_hour"],
            rel=1e-6,
        )

    def test_invalid_tonnes(self):
        """Non-positive tonnes should raise ValueError."""
        with pytest.raises(ValueError, match="tonnes_mined"):
            dig_rate_survey(0, 10, 250)

    def test_invalid_hours(self):
        """Non-positive hours should raise ValueError."""
        with pytest.raises(ValueError, match="operating_hours"):
            dig_rate_survey(5000, 0, 250)
