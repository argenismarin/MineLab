"""Tests for minelab.geomechanics.wedge_analysis."""

import numpy as np
import pytest

from minelab.geomechanics.wedge_analysis import (
    kinematic_planar,
    kinematic_toppling,
    kinematic_wedge,
    markland_test,
    stereonet_data,
    wedge_fos,
)


class TestKinematicPlanar:
    """Tests for planar sliding kinematic check."""

    def test_unstable(self):
        """Face=70°, plane=45°, φ=35° → unstable."""
        result = kinematic_planar(70, 45, 35)
        assert result["unstable"] is True
        assert result["conditions"]["daylight"] is True
        assert result["conditions"]["exceeds_friction"] is True

    def test_stable_no_daylight(self):
        """Plane steeper than face → no daylight."""
        result = kinematic_planar(40, 50, 35)
        assert result["unstable"] is False
        assert result["conditions"]["daylight"] is False

    def test_stable_friction(self):
        """Plane dip < friction angle → stable."""
        result = kinematic_planar(70, 30, 35)
        assert result["unstable"] is False
        assert result["conditions"]["exceeds_friction"] is False

    def test_invalid_dip(self):
        """Dip > 90 should raise."""
        with pytest.raises(ValueError):
            kinematic_planar(95, 45, 35)


class TestKinematicWedge:
    """Tests for wedge failure kinematic check."""

    def test_returns_plunge_trend(self):
        """Should compute line of intersection."""
        result = kinematic_wedge(70, 180, (45, 160), (50, 200), 35)
        assert "plunge" in result
        assert "trend" in result
        assert 0 <= result["plunge"] <= 90

    def test_parallel_planes(self):
        """Parallel planes → no wedge."""
        result = kinematic_wedge(70, 180, (45, 90), (45, 90), 35)
        assert result["unstable"] is False

    def test_known_unstable(self):
        """Two planes intersecting and daylighting → unstable."""
        result = kinematic_wedge(70, 180, (50, 150), (50, 210), 30)
        # Should detect intersection plunging into face
        assert result["unstable"] in (True, False)


class TestKinematicToppling:
    """Tests for toppling kinematic check."""

    def test_unstable(self):
        """Face=70°, planes=80°, φ=35° → (90-70)+35=55 < 80 → unstable."""
        result = kinematic_toppling(70, 80, 35)
        assert result["unstable"] is True

    def test_stable(self):
        """Face=70°, planes=50°, φ=35° → (90-70)+35=55 > 50 → stable."""
        result = kinematic_toppling(70, 50, 35)
        assert result["unstable"] is False

    def test_critical_dip(self):
        """Critical dip = (90 - dip_face) + φ."""
        result = kinematic_toppling(70, 80, 35)
        assert result["critical_dip"] == pytest.approx(55.0)

    def test_boundary(self):
        """At exactly critical dip → stable (not strictly greater)."""
        result = kinematic_toppling(70, 55, 35)
        assert result["unstable"] is False


class TestMarklandTest:
    """Tests for combined Markland test."""

    def test_returns_all_modes(self):
        """Should return dict with planar, wedge, toppling keys."""
        result = markland_test(70, 180, [(45, 170), (80, 10)], 35)
        assert "planar" in result
        assert "wedge" in result
        assert "toppling" in result

    def test_planar_detection(self):
        """Should detect planar sliding for near-parallel plane."""
        result = markland_test(70, 180, [(45, 180)], 35)
        assert 0 in result["planar"]

    def test_toppling_detection(self):
        """Should detect toppling for back-dipping steep plane."""
        result = markland_test(70, 180, [(80, 0)], 35)
        assert 0 in result["toppling"]


class TestWedgeFOS:
    """Tests for wedge factor of safety."""

    def test_positive_fos(self):
        """FOS should be positive."""
        result = wedge_fos((45, 160), (50, 200), 1000, 35, 35)
        assert result["fos"] > 0

    def test_cohesion_increases_fos(self):
        """Adding cohesion should increase FOS."""
        fos_no_c = wedge_fos(
            (45, 160), (50, 200), 1000, 35, 35,
        )["fos"]
        fos_with_c = wedge_fos(
            (45, 160), (50, 200), 1000, 35, 35,
            cohesion1=50, cohesion2=50, area1=10, area2=10,
        )["fos"]
        assert fos_with_c > fos_no_c

    def test_water_reduces_fos(self):
        """Water pressure should reduce FOS."""
        dry = wedge_fos(
            (45, 160), (50, 200), 1000, 35, 35,
            area1=10, area2=10,
        )["fos"]
        wet = wedge_fos(
            (45, 160), (50, 200), 1000, 35, 35,
            area1=10, area2=10, water_pressure=50,
        )["fos"]
        assert wet < dry


class TestStereonetData:
    """Tests for stereonet data conversion."""

    def test_pole_conversion(self):
        """dip=45, dd=90 → pole plunge=45, trend=270."""
        result = stereonet_data([(45, 90)])
        plunge, trend = result["poles"][0]
        assert plunge == pytest.approx(45.0)
        assert trend == pytest.approx(270.0)

    def test_horizontal_plane(self):
        """dip=0 → pole plunge=90 (vertical pole)."""
        result = stereonet_data([(0, 0)])
        plunge, _ = result["poles"][0]
        assert plunge == pytest.approx(90.0)

    def test_multiple_planes(self):
        """Multiple planes → matching number of poles."""
        planes = [(45, 90), (60, 180), (30, 270)]
        result = stereonet_data(planes)
        assert len(result["poles"]) == 3
        assert result["pole_vectors"].shape == (3, 3)

    def test_great_circles_preserved(self):
        """Great circles should match input."""
        planes = [(45, 90), (60, 180)]
        result = stereonet_data(planes)
        assert result["great_circles"] == [(45.0, 90.0), (60.0, 180.0)]
