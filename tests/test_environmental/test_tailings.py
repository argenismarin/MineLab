"""Tests for minelab.environmental.tailings."""

import pytest

from minelab.environmental.tailings import (
    tailings_beach_angle,
    tailings_storage_capacity,
)


class TestTailingsStorageCapacity:
    """Tests for TSF volume calculation."""

    def test_positive_volume(self):
        """Volume should be positive."""
        result = tailings_storage_capacity(50000, 20, 2.0)
        assert result["volume"] > 0

    def test_larger_area_more_capacity(self):
        """Larger area → more capacity."""
        r_small = tailings_storage_capacity(10000, 20, 2.0)
        r_large = tailings_storage_capacity(50000, 20, 2.0)
        assert r_large["volume"] > r_small["volume"]


class TestTailingsBeachAngle:
    """Tests for beach angle estimation."""

    def test_positive(self):
        """Beach angle should be positive."""
        angle = tailings_beach_angle(0.5, 50)
        assert angle > 0

    def test_typical_range(self):
        """Typical 0.5-5 degrees."""
        angle = tailings_beach_angle(0.6, 75)
        assert 0 < angle < 10
