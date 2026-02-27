"""Tests for minelab.mine_planning.mine_design."""

import pytest

from minelab.mine_planning.mine_design import (
    pit_geometry,
    pit_volume_tonnage,
    ramp_design,
)


class TestPitGeometry:
    """Tests for pit geometry calculations."""

    def test_known_value(self):
        """H=10, berm=8, face=75° → IRA ≈ 43°.

        IRA = atan(H / (H/tan(face) + berm))
            = atan(10 / (10/tan(75°) + 8))
            = atan(10 / (2.679 + 8))
            = atan(10 / 10.679) ≈ 43.1°
        Ref: Hustrulid et al. 2013.
        """
        result = pit_geometry(10, 8, 75)
        assert result["inter_ramp_angle"] == pytest.approx(43.12, abs=1)

    def test_steeper_face_steeper_ira(self):
        """Steeper face → steeper IRA."""
        r_shallow = pit_geometry(10, 8, 60)
        r_steep = pit_geometry(10, 8, 80)
        assert r_steep["inter_ramp_angle"] > r_shallow["inter_ramp_angle"]


class TestRampDesign:
    """Tests for ramp design."""

    def test_returns_params(self):
        """Should return design parameters."""
        result = ramp_design(25, 10, 30)
        assert "effective_width" in result or "width" in result

    def test_positive_values(self):
        """All values should be positive."""
        result = ramp_design(25, 10, 30)
        for v in result.values():
            if isinstance(v, (int, float)):
                assert v >= 0


class TestPitVolumeTonnage:
    """Tests for pit volume and tonnage."""

    def test_positive_volume(self):
        """Volume should be positive."""
        areas = [10000, 8000, 6000, 4000]
        result = pit_volume_tonnage(areas, 10, 2.7)
        assert result["total_volume"] > 0

    def test_tonnage_equals_volume_times_density(self):
        """Tonnage = volume * density."""
        areas = [10000, 8000, 6000]
        result = pit_volume_tonnage(areas, 10, 2.7)
        assert result["total_tonnage"] == pytest.approx(
            result["total_volume"] * 2.7, rel=0.01
        )
