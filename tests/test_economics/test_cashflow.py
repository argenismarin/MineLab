"""Tests for minelab.economics.cashflow module."""

import math

import pytest

from minelab.economics.cashflow import (
    discounted_payback,
    equivalent_annual_annuity,
    irr,
    npv,
    payback_period,
    profitability_index,
)


# -------------------------------------------------------------------------
# NPV
# -------------------------------------------------------------------------


class TestNPV:
    """Tests for the Net Present Value function."""

    def test_known_value(self):
        """NPV(10%, [-1000, 300, 420, 680]) = 130.73 (sum of discounted CFs from t=0)."""
        result = npv(0.10, [-1000, 300, 420, 680])
        assert result == pytest.approx(130.7288, rel=1e-4)

    def test_zero_rate(self):
        """At 0% discount rate, NPV equals the arithmetic sum."""
        cfs = [-1000, 300, 420, 680]
        assert npv(0.0, cfs) == pytest.approx(sum(cfs))

    def test_single_cashflow(self):
        """NPV of a single cash flow at t=0 is the cash flow itself."""
        assert npv(0.10, [500]) == pytest.approx(500.0)

    def test_all_negative(self):
        """NPV of all-negative flows is negative."""
        assert npv(0.10, [-100, -200, -300]) < 0

    def test_rate_minus_one_raises(self):
        """Rate <= -1 should raise ValueError."""
        with pytest.raises(ValueError, match="greater than -1"):
            npv(-1.0, [-1000, 500])

    def test_sample_cashflows_fixture(self, sample_cashflows):
        """NPV with fixture data should be a valid number."""
        result = npv(0.10, sample_cashflows)
        assert math.isfinite(result)


# -------------------------------------------------------------------------
# IRR
# -------------------------------------------------------------------------


class TestIRR:
    """Tests for the Internal Rate of Return function."""

    def test_known_value(self):
        """IRR of [-1000, 300, 420, 680] ~ 16.34%."""
        result = irr([-1000, 300, 420, 680])
        assert result == pytest.approx(0.1634, rel=1e-2)

    def test_npv_at_irr_is_zero(self):
        """NPV evaluated at IRR should be approximately zero."""
        cfs = [-1000, 300, 420, 680]
        r = irr(cfs)
        assert npv(r, cfs) == pytest.approx(0.0, abs=1e-8)

    def test_no_sign_change_raises(self):
        """Cash flows with no sign change should raise ValueError."""
        with pytest.raises(ValueError, match="sign change"):
            irr([100, 200, 300])

    def test_simple_irr(self):
        """IRR of [-100, 110] should be 10%."""
        result = irr([-100, 110])
        assert result == pytest.approx(0.10, rel=1e-4)


# -------------------------------------------------------------------------
# Payback Period
# -------------------------------------------------------------------------


class TestPaybackPeriod:
    """Tests for the simple payback period function."""

    def test_known_value(self):
        """Payback of [-1000, 300, 420, 680]: cumulative at t=2 is -280, at t=3 is 400."""
        result = payback_period([-1000, 300, 420, 680])
        # frac = 280 / 680 = 0.4118
        assert result == pytest.approx(2.4118, rel=1e-3)

    def test_immediate_payback(self):
        """If first cash flow is non-negative, payback is 0."""
        assert payback_period([100, 200]) == 0.0

    def test_never_recovered(self):
        """If cumulative never reaches 0, return inf."""
        assert payback_period([-1000, 100, 100]) == float("inf")

    def test_exact_period(self):
        """Payback at exact period boundary."""
        result = payback_period([-100, 50, 50])
        assert result == pytest.approx(2.0)


# -------------------------------------------------------------------------
# Discounted Payback
# -------------------------------------------------------------------------


class TestDiscountedPayback:
    """Tests for the discounted payback period function."""

    def test_longer_than_simple(self):
        """Discounted payback should be >= simple payback for positive rate."""
        cfs = [-1000, 300, 420, 680]
        simple = payback_period(cfs)
        disc = discounted_payback(0.10, cfs)
        assert disc >= simple

    def test_known_value(self):
        """Discounted payback of [-1000, 300, 420, 680] at 10%."""
        result = discounted_payback(0.10, [-1000, 300, 420, 680])
        assert result == pytest.approx(2.7441, rel=1e-2)

    def test_zero_rate_equals_simple(self):
        """At 0% rate, discounted payback equals simple payback."""
        cfs = [-1000, 300, 420, 680]
        assert discounted_payback(0.0, cfs) == pytest.approx(payback_period(cfs))

    def test_never_recovered(self):
        """High discount rate prevents recovery."""
        assert discounted_payback(5.0, [-1000, 100, 100, 100]) == float("inf")

    def test_rate_minus_one_raises(self):
        with pytest.raises(ValueError):
            discounted_payback(-1.0, [-1000, 500])


# -------------------------------------------------------------------------
# Profitability Index
# -------------------------------------------------------------------------


class TestProfitabilityIndex:
    """Tests for the profitability index function."""

    def test_known_value(self):
        """PI of [-1000, 300, 420, 680] at 10% ~ 1.1307."""
        result = profitability_index(0.10, [-1000, 300, 420, 680])
        assert result == pytest.approx(1.1307, rel=1e-3)

    def test_pi_greater_than_one_positive_npv(self):
        """If NPV > 0, PI should be > 1."""
        cfs = [-1000, 300, 420, 680]
        assert profitability_index(0.10, cfs) > 1.0

    def test_zero_investment_raises(self):
        """Zero initial investment should raise ValueError."""
        with pytest.raises(ValueError, match="must not be zero"):
            profitability_index(0.10, [0, 300, 420])


# -------------------------------------------------------------------------
# Equivalent Annual Annuity
# -------------------------------------------------------------------------


class TestEquivalentAnnualAnnuity:
    """Tests for the EAA function."""

    def test_known_value(self):
        """EAA(10%, 178.77, 3) ~ 71.89."""
        result = equivalent_annual_annuity(0.10, 178.77, 3)
        assert result == pytest.approx(71.89, rel=1e-2)

    def test_single_year(self):
        """For 1 year, EAA = NPV * (1+r)."""
        result = equivalent_annual_annuity(0.10, 100, 1)
        # EAA = 100 * 0.10 / (1 - 1/1.10) = 100 * 0.10 / 0.0909... = 110
        assert result == pytest.approx(110.0, rel=1e-3)

    def test_zero_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            equivalent_annual_annuity(0.0, 100, 5)

    def test_zero_years_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            equivalent_annual_annuity(0.10, 100, 0)
