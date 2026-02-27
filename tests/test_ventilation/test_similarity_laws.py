"""Tests for minelab.ventilation.similarity_laws."""

import pytest

from minelab.ventilation.similarity_laws import (
    fan_affinity_laws,
    specific_speed,
)


class TestFanAffinityLaws:
    """Tests for fan affinity laws."""

    def test_double_speed(self):
        """Double speed → 2x Q, 4x P, 8x Power."""
        result = fan_affinity_laws(50, 2000, 150000, 1000, 2000)
        assert result["Q2"] == pytest.approx(100, rel=0.01)
        assert result["P2"] == pytest.approx(8000, rel=0.01)
        assert result["Power2"] == pytest.approx(1200000, rel=0.01)

    def test_same_speed(self):
        """Same speed → same values."""
        result = fan_affinity_laws(50, 2000, 150000, 1000, 1000)
        assert result["Q2"] == pytest.approx(50, rel=0.01)
        assert result["P2"] == pytest.approx(2000, rel=0.01)

    def test_different_diameter(self):
        """Double diameter, same speed → 8x Q, 4x P, 32x Power."""
        result = fan_affinity_laws(50, 2000, 150000, 1000, 1000, D1=1.0, D2=2.0)
        assert result["Q2"] == pytest.approx(400, rel=0.01)
        assert result["P2"] == pytest.approx(8000, rel=0.01)


class TestSpecificSpeed:
    """Tests for specific speed."""

    def test_positive(self):
        """Should be positive."""
        ns = specific_speed(1500, 50, 2000)
        assert ns > 0

    def test_known(self):
        """Known calculation."""
        # N = 1500/60 = 25 rev/s, Q=50, P=2000
        # Ns = 25 * 50^0.5 / 2000^0.75
        import math
        expected = 25 * math.sqrt(50) / (2000 ** 0.75)
        ns = specific_speed(1500, 50, 2000)
        assert ns == pytest.approx(expected, rel=0.01)
