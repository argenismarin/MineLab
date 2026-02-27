"""Tests for minelab.drilling_blasting.blastability."""

import pytest

from minelab.drilling_blasting.blastability import (
    lilly_blastability_index,
    rock_factor_from_bi,
)


class TestLillyBlastabilityIndex:
    """Tests for Lilly BI."""

    def test_known_value(self):
        """BI = 0.5 * (RMD + JF + JPS + RDI + HF)."""
        bi = lilly_blastability_index(20, 30, 20, 25, 10)
        assert bi == pytest.approx(0.5 * (20 + 30 + 20 + 25 + 10), rel=0.01)

    def test_positive(self):
        """BI should be positive."""
        bi = lilly_blastability_index(10, 20, 10, 15, 5)
        assert bi > 0


class TestRockFactorFromBI:
    """Tests for rock factor from BI."""

    def test_known_value(self):
        """A = 0.06 * BI."""
        a = rock_factor_from_bi(50)
        assert a == pytest.approx(3.0, rel=0.01)

    def test_positive(self):
        """A should be positive."""
        a = rock_factor_from_bi(30)
        assert a > 0
