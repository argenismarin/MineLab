"""Tests for minelab.data_management.desurvey module."""

import numpy as np
import pandas as pd
import pytest

from minelab.data_management.desurvey import (
    balanced_tangential,
    compute_coordinates,
    minimum_curvature,
    tangential,
)


@pytest.fixture
def vertical_survey():
    """Single vertical hole, dip = -90 (straight down)."""
    return pd.DataFrame({
        "hole_id": ["DH1"] * 6,
        "depth": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        "azimuth": [0.0] * 6,
        "dip": [-90.0] * 6,
    })


@pytest.fixture
def deviated_survey():
    """Hole that starts vertical and gradually deviates to 45-deg dip east."""
    return pd.DataFrame({
        "hole_id": ["DH2"] * 4,
        "depth": [0.0, 50.0, 100.0, 150.0],
        "azimuth": [90.0, 90.0, 90.0, 90.0],  # east
        "dip": [-90.0, -70.0, -50.0, -45.0],
    })


@pytest.fixture
def multi_hole_survey():
    """Two vertical holes."""
    return pd.DataFrame({
        "hole_id": ["DH1", "DH1", "DH2", "DH2"],
        "depth": [0.0, 100.0, 0.0, 50.0],
        "azimuth": [0.0, 0.0, 90.0, 90.0],
        "dip": [-90.0, -90.0, -90.0, -90.0],
    })


class TestMinimumCurvatureVertical:
    """Test minimum curvature with a vertical hole."""

    def test_vertical_dz(self, vertical_survey):
        """Vertical hole: all displacement should be in dz."""
        result = minimum_curvature(vertical_survey)
        assert len(result) == 6
        # dx and dy should be ~0
        np.testing.assert_allclose(result["dx"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dy"].values, 0.0, atol=1e-10)
        # dz should equal depth (vertical down = dz accumulation)
        np.testing.assert_allclose(result["dz"].values, result["depth"].values, atol=1e-10)

    def test_vertical_last_point(self, vertical_survey):
        """At 100m depth, dz should be 100."""
        result = minimum_curvature(vertical_survey)
        last = result.iloc[-1]
        np.testing.assert_allclose(float(last["dz"]), 100.0, atol=1e-10)


class TestMinimumCurvatureDeviated:
    """Test minimum curvature with a deviated hole."""

    def test_deviated_has_east_displacement(self, deviated_survey):
        """Hole deviating east should show positive dx."""
        result = minimum_curvature(deviated_survey)
        # dx should increase as hole deviates east
        assert float(result.iloc[-1]["dx"]) > 0

    def test_deviated_dz_less_than_md(self, deviated_survey):
        """Deviated hole: vertical depth < measured depth."""
        result = minimum_curvature(deviated_survey)
        last_dz = float(result.iloc[-1]["dz"])
        last_md = float(result.iloc[-1]["depth"])
        assert last_dz < last_md

    def test_deviated_total_distance(self, deviated_survey):
        """The total 3D distance between consecutive points should
        approximate the measured depth intervals."""
        result = minimum_curvature(deviated_survey)
        total_3d = 0.0
        for i in range(1, len(result)):
            ddx = float(result.iloc[i]["dx"] - result.iloc[i - 1]["dx"])
            ddy = float(result.iloc[i]["dy"] - result.iloc[i - 1]["dy"])
            ddz = float(result.iloc[i]["dz"] - result.iloc[i - 1]["dz"])
            total_3d += np.sqrt(ddx**2 + ddy**2 + ddz**2)
        total_md = float(result.iloc[-1]["depth"])
        # 3D path length should closely match measured depth
        np.testing.assert_allclose(total_3d, total_md, rtol=0.01)


class TestTangential:
    """Test tangential method."""

    def test_vertical(self, vertical_survey):
        result = tangential(vertical_survey)
        np.testing.assert_allclose(result["dx"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dy"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dz"].values, result["depth"].values, atol=1e-10)


class TestBalancedTangential:
    """Test balanced tangential method."""

    def test_vertical(self, vertical_survey):
        result = balanced_tangential(vertical_survey)
        np.testing.assert_allclose(result["dx"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dy"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dz"].values, result["depth"].values, atol=1e-10)

    def test_balanced_between_tangential_and_mincurv(self, deviated_survey):
        """Balanced tangential result should differ from tangential."""
        tang = tangential(deviated_survey)
        bal = balanced_tangential(deviated_survey)
        # They should not be identical for a deviated hole
        assert not np.allclose(tang["dx"].values, bal["dx"].values)


class TestComputeCoordinates:
    """Test absolute coordinate computation."""

    def test_vertical_coordinates(self, vertical_survey):
        desurvey = minimum_curvature(vertical_survey)
        coords = compute_coordinates(1000.0, 2000.0, 500.0, desurvey)
        # At surface: x=1000, y=2000, z=500
        np.testing.assert_allclose(float(coords.iloc[0]["x"]), 1000.0)
        np.testing.assert_allclose(float(coords.iloc[0]["y"]), 2000.0)
        np.testing.assert_allclose(float(coords.iloc[0]["z"]), 500.0)
        # At 100m depth: z = 500 - 100 = 400
        np.testing.assert_allclose(float(coords.iloc[-1]["z"]), 400.0, atol=1e-10)
        np.testing.assert_allclose(float(coords.iloc[-1]["x"]), 1000.0, atol=1e-10)

    def test_deviated_coordinates(self, deviated_survey):
        desurvey = minimum_curvature(deviated_survey)
        coords = compute_coordinates(0.0, 0.0, 0.0, desurvey)
        # Elevation should be negative (below collar)
        assert float(coords.iloc[-1]["z"]) < 0.0
        # Easting should be positive (deviation to east)
        assert float(coords.iloc[-1]["x"]) > 0.0


class TestMultipleHoles:
    """Test that multi-hole DataFrames are handled correctly."""

    def test_multi_hole_minimum_curvature(self, multi_hole_survey):
        result = minimum_curvature(multi_hole_survey)
        dh1 = result[result["hole_id"] == "DH1"]
        dh2 = result[result["hole_id"] == "DH2"]
        assert len(dh1) == 2
        assert len(dh2) == 2
        np.testing.assert_allclose(float(dh1.iloc[-1]["dz"]), 100.0, atol=1e-10)
        np.testing.assert_allclose(float(dh2.iloc[-1]["dz"]), 50.0, atol=1e-10)


class TestEmptyInput:
    """Test with empty DataFrame."""

    def test_empty_minimum_curvature(self):
        empty = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        result = minimum_curvature(empty)
        assert "dx" in result.columns

    def test_empty_tangential(self):
        empty = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        result = tangential(empty)
        assert "dx" in result.columns

    def test_empty_balanced(self):
        empty = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        result = balanced_tangential(empty)
        assert "dx" in result.columns
