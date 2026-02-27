"""Tests for minelab.economics.revenue module."""

import pytest

from minelab.economics.revenue import (
    cutoff_grade_breakeven,
    gross_revenue,
    net_smelter_return,
)


# -------------------------------------------------------------------------
# Gross Revenue
# -------------------------------------------------------------------------


class TestGrossRevenue:
    """Tests for the gross_revenue function."""

    def test_basic(self):
        """1 Mt at 1.5 g/t Au, $60/g, 90% recovery."""
        result = gross_revenue(1_000_000, 1.5, 60.0, recovery=0.90)
        assert result == pytest.approx(81_000_000.0)

    def test_full_recovery(self):
        """Default recovery = 1.0."""
        result = gross_revenue(1000, 2.0, 50.0)
        assert result == pytest.approx(100_000.0)

    def test_zero_tonnage(self):
        assert gross_revenue(0, 1.5, 60.0) == pytest.approx(0.0)

    def test_zero_grade(self):
        assert gross_revenue(1000, 0.0, 60.0) == pytest.approx(0.0)

    def test_negative_tonnage_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gross_revenue(-1, 1.5, 60.0)

    def test_invalid_recovery_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            gross_revenue(1000, 1.5, 60.0, recovery=1.5)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gross_revenue(1000, 1.5, -10)


# -------------------------------------------------------------------------
# Net Smelter Return
# -------------------------------------------------------------------------


class TestNetSmelterReturn:
    """Tests for the net_smelter_return function."""

    def test_basic(self):
        """NSR = 1M * 0.95 - 50k - 20k - 5k = 875k."""
        result = net_smelter_return(
            1_000_000, tc=50_000, rc=20_000, penalties=5_000, payable_pct=0.95
        )
        assert result == pytest.approx(875_000.0)

    def test_no_deductions(self):
        """Without charges, NSR = gross revenue."""
        result = net_smelter_return(1_000_000, tc=0, rc=0)
        assert result == pytest.approx(1_000_000.0)

    def test_full_payable(self):
        """Default payable_pct=1.0."""
        result = net_smelter_return(500_000, tc=10_000, rc=5_000)
        assert result == pytest.approx(485_000.0)

    def test_negative_tc_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            net_smelter_return(1_000_000, tc=-1, rc=0)

    def test_invalid_payable_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            net_smelter_return(1_000_000, tc=0, rc=0, payable_pct=1.5)

    def test_nsr_can_be_negative(self):
        """NSR can be negative if charges exceed payable revenue."""
        result = net_smelter_return(100, tc=50, rc=50, penalties=50)
        assert result < 0


# -------------------------------------------------------------------------
# Cut-Off Grade
# -------------------------------------------------------------------------


class TestCutoffGradeBreakeven:
    """Tests for the breakeven cut-off grade function."""

    def test_basic(self):
        """COG = 30 / (60 * 0.90) = 0.5556."""
        result = cutoff_grade_breakeven(60.0, 0.90, 30.0)
        assert result == pytest.approx(0.5556, rel=1e-3)

    def test_perfect_recovery(self):
        """COG = cost / price when recovery = 1."""
        result = cutoff_grade_breakeven(50.0, 1.0, 25.0)
        assert result == pytest.approx(0.50)

    def test_zero_cost(self):
        """Zero cost -> zero COG."""
        assert cutoff_grade_breakeven(50.0, 0.90, 0.0) == pytest.approx(0.0)

    def test_higher_cost_higher_cog(self):
        """Higher costs should yield a higher cut-off grade."""
        cog1 = cutoff_grade_breakeven(60.0, 0.90, 20.0)
        cog2 = cutoff_grade_breakeven(60.0, 0.90, 40.0)
        assert cog2 > cog1

    def test_zero_price_raises(self):
        with pytest.raises(ValueError, match="positive"):
            cutoff_grade_breakeven(0, 0.90, 30.0)

    def test_zero_recovery_raises(self):
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            cutoff_grade_breakeven(60.0, 0.0, 30.0)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cutoff_grade_breakeven(60.0, 0.90, -1)
