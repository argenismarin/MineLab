"""Tests for minelab.mine_planning.cutoff_grade."""

import pytest

from minelab.mine_planning.cutoff_grade import (
    breakeven_cutoff,
    lane_cutoff,
    marginal_cutoff,
)


class TestBreakevenCutoff:
    """Tests for breakeven cutoff grade."""

    def test_known_value(self):
        """Known Cu cutoff."""
        # COG = (proc + mine + ga) / (price * recovery)
        # = (15 + 3 + 2) / (8000 * 0.9) = 20/7200 ≈ 0.00278
        cog = breakeven_cutoff(8000, 0.9, 15, 3, 2)
        assert cog == pytest.approx(0.00278, rel=0.02)

    def test_higher_price_lower_cutoff(self):
        """Higher price → lower cutoff."""
        c_low = breakeven_cutoff(8000, 0.9, 15, 3)
        c_high = breakeven_cutoff(12000, 0.9, 15, 3)
        assert c_high < c_low

    def test_positive(self):
        """Cutoff should be positive."""
        cog = breakeven_cutoff(8000, 0.9, 15, 3)
        assert cog > 0


class TestLaneCutoff:
    """Tests for Lane's three cutoffs."""

    def test_returns_three_cutoffs(self):
        """Should return mine, mill, refinery cutoffs."""
        costs = {"mining": 3, "processing": 15, "ga": 2}
        result = lane_cutoff(50000, 10000, 5000, costs, 8000, 0.9)
        assert "g_mine" in result
        assert "g_mill" in result
        assert "g_optimum" in result


class TestMarginalCutoff:
    """Tests for marginal cutoff grade."""

    def test_known_value(self):
        """COG_marginal = proc / (price * recovery)."""
        cog = marginal_cutoff(8000, 0.9, 15)
        expected = 15 / (8000 * 0.9)
        assert cog == pytest.approx(expected, rel=0.01)

    def test_lower_than_breakeven(self):
        """Marginal < breakeven."""
        marg = marginal_cutoff(8000, 0.9, 15)
        be = breakeven_cutoff(8000, 0.9, 15, 3, 2)
        assert marg < be
