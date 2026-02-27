"""Tests for minelab.economics.cost_models module."""

import pytest

from minelab.economics.cost_models import (
    capex_estimate,
    depreciation_declining_balance,
    depreciation_straight_line,
    opex_per_tonne,
    stripping_cost,
    taylor_rule,
)


# -------------------------------------------------------------------------
# CAPEX — Six-Tenths Rule
# -------------------------------------------------------------------------


class TestCapexEstimate:
    """Tests for the six-tenths rule capital cost estimator."""

    def test_known_value(self):
        """Scale 2000 t/d plant ($10M) to 5000 t/d with exponent=0.6."""
        result = capex_estimate(5000, 10_000_000, 2000, exponent=0.6)
        # (5000/2000)^0.6 = 2.5^0.6 = 1.7329...
        assert result == pytest.approx(17_328_621.08, rel=1e-3)

    def test_same_capacity(self):
        """If capacity equals base capacity, cost equals base cost."""
        assert capex_estimate(2000, 10_000_000, 2000) == pytest.approx(10_000_000)

    def test_exponent_one(self):
        """With exponent=1.0, scaling is linear."""
        result = capex_estimate(4000, 10_000_000, 2000, exponent=1.0)
        assert result == pytest.approx(20_000_000)

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            capex_estimate(0, 10_000_000, 2000)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="positive"):
            capex_estimate(5000, -1, 2000)


# -------------------------------------------------------------------------
# OPEX Per Tonne
# -------------------------------------------------------------------------


class TestOpexPerTonne:
    """Tests for operating cost summation."""

    def test_basic(self):
        assert opex_per_tonne(2.50, 8.00, 1.50) == pytest.approx(12.0)

    def test_with_other(self):
        assert opex_per_tonne(2.50, 8.00, 1.50, other=0.50) == pytest.approx(12.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            opex_per_tonne(-1, 8.00, 1.50)


# -------------------------------------------------------------------------
# Stripping Cost
# -------------------------------------------------------------------------


class TestStrippingCost:
    """Tests for stripping cost allocation."""

    def test_basic(self):
        """3:1 strip ratio at $2.50/t waste -> $7.50/t ore."""
        result = stripping_cost(3_000_000, 1_000_000, 2.50)
        assert result == pytest.approx(7.50)

    def test_zero_waste(self):
        assert stripping_cost(0, 1_000_000, 2.50) == pytest.approx(0.0)

    def test_zero_ore_raises(self):
        with pytest.raises(ValueError, match="positive"):
            stripping_cost(1000, 0, 2.50)


# -------------------------------------------------------------------------
# Depreciation — Straight Line
# -------------------------------------------------------------------------


class TestDepreciationStraightLine:
    """Tests for straight-line depreciation."""

    def test_basic(self):
        result = depreciation_straight_line(1_000_000, 100_000, 10)
        assert result == pytest.approx(90_000.0)

    def test_zero_salvage(self):
        result = depreciation_straight_line(500_000, 0, 5)
        assert result == pytest.approx(100_000.0)

    def test_salvage_exceeds_capex_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            depreciation_straight_line(100_000, 200_000, 10)

    def test_zero_life_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            depreciation_straight_line(100_000, 0, 0)


# -------------------------------------------------------------------------
# Depreciation — Declining Balance
# -------------------------------------------------------------------------


class TestDepreciationDecliningBalance:
    """Tests for declining-balance depreciation."""

    def test_basic(self):
        result = depreciation_declining_balance(1_000_000, 0.20, 3)
        assert result == pytest.approx([200_000, 160_000, 128_000])

    def test_length(self):
        result = depreciation_declining_balance(1_000_000, 0.20, 5)
        assert len(result) == 5

    def test_sum_less_than_capex(self):
        """Total depreciation should never exceed CAPEX."""
        result = depreciation_declining_balance(1_000_000, 0.30, 10)
        assert sum(result) < 1_000_000

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            depreciation_declining_balance(1_000_000, 0.0, 5)

    def test_rate_over_one_raises(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            depreciation_declining_balance(1_000_000, 1.5, 5)


# -------------------------------------------------------------------------
# Taylor's Rule
# -------------------------------------------------------------------------


class TestTaylorRule:
    """Tests for Taylor's Rule mine capacity estimation."""

    def test_known_value(self):
        """Taylor's rule: 0.25 * 100^0.75 = 0.25 * 31.623 = 7.906."""
        result = taylor_rule(100)
        assert result == pytest.approx(7.9057, rel=1e-3)

    def test_small_deposit(self):
        """1 Mt deposit -> 0.25 Mt/yr."""
        result = taylor_rule(1)
        assert result == pytest.approx(0.25, rel=1e-4)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            taylor_rule(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            taylor_rule(-10)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestStrippingCostValidation:
    """Additional validation tests for stripping_cost."""

    def test_negative_waste_tonnes_raises(self):
        """Negative waste tonnes should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            stripping_cost(-100, 1_000_000, 2.50)

    def test_negative_cost_per_tonne_raises(self):
        """Negative cost per tonne should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            stripping_cost(1000, 1_000_000, -1.0)


class TestDepreciationStraightLineValidation:
    """Additional validation tests for depreciation_straight_line."""

    def test_zero_capex_raises(self):
        """Zero CAPEX should raise ValueError."""
        with pytest.raises(ValueError, match="CAPEX must be positive"):
            depreciation_straight_line(0, 0, 10)

    def test_negative_capex_raises(self):
        """Negative CAPEX should raise ValueError."""
        with pytest.raises(ValueError, match="CAPEX must be positive"):
            depreciation_straight_line(-100, 0, 10)

    def test_negative_salvage_raises(self):
        """Negative salvage value should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            depreciation_straight_line(100_000, -1, 10)


class TestDepreciationDecliningBalanceValidation:
    """Additional validation tests for depreciation_declining_balance."""

    def test_zero_capex_raises(self):
        """Zero CAPEX should raise ValueError."""
        with pytest.raises(ValueError, match="CAPEX must be positive"):
            depreciation_declining_balance(0, 0.20, 3)

    def test_negative_capex_raises(self):
        """Negative CAPEX should raise ValueError."""
        with pytest.raises(ValueError, match="CAPEX must be positive"):
            depreciation_declining_balance(-500, 0.20, 3)

    def test_zero_years_raises(self):
        """Zero years should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            depreciation_declining_balance(1_000_000, 0.20, 0)
