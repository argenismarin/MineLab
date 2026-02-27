"""Tests for minelab.drilling_blasting.flyrock."""

import pytest

from minelab.drilling_blasting.flyrock import (
    flyrock_range,
    safety_distance,
)


class TestFlyrockRange:
    """Tests for flyrock range estimation."""

    def test_positive(self):
        """Range should be positive."""
        r = flyrock_range(89, 3.0, 2.1, 5.0)
        assert r > 0

    def test_less_stemming_more_flyrock(self):
        """Less stemming → more flyrock."""
        r_good = flyrock_range(89, 3.0, 2.5, 5.0)
        r_bad = flyrock_range(89, 3.0, 1.0, 5.0)
        assert r_bad > r_good


class TestSafetyDistance:
    """Tests for safety distance."""

    def test_default_factor(self):
        """D_safe = range * 1.5."""
        d = safety_distance(200)
        assert d == pytest.approx(300, rel=0.01)

    def test_custom_factor(self):
        """D_safe = range * factor."""
        d = safety_distance(200, factor=2.0)
        assert d == pytest.approx(400, rel=0.01)
