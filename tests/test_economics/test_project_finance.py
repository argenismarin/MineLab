"""Tests for minelab.economics.project_finance."""

import pytest

from minelab.economics.project_finance import (
    break_even_metal_price,
    debt_service_coverage_ratio,
    leverage_effect_irr,
    loan_amortization,
    working_capital_requirement,
)


class TestDebtServiceCoverageRatio:
    """Tests for debt_service_coverage_ratio."""

    def test_known_value(self):
        """DSCR = 15M / 8M = 1.875."""
        result = debt_service_coverage_ratio(15_000_000, 8_000_000)
        assert result == pytest.approx(1.875, rel=1e-4)

    def test_ratio_one(self):
        """Equal EBITDA and debt service gives ratio of 1."""
        result = debt_service_coverage_ratio(5_000_000, 5_000_000)
        assert result == pytest.approx(1.0)

    def test_zero_ebitda(self):
        """Zero EBITDA gives zero DSCR."""
        result = debt_service_coverage_ratio(0, 5_000_000)
        assert result == pytest.approx(0.0)

    def test_invalid_debt_service(self):
        """Non-positive debt service should raise ValueError."""
        with pytest.raises(ValueError, match="annual_debt_service"):
            debt_service_coverage_ratio(15_000_000, 0)


class TestLoanAmortization:
    """Tests for loan_amortization."""

    def test_known_value(self):
        """Verify annual payment for $1M at 8% for 5 years."""
        r = 0.08
        n = 5
        factor = (1 + r) ** n
        expected_pmt = 1_000_000 * r * factor / (factor - 1)
        result = loan_amortization(1_000_000, 0.08, 5)
        assert result["annual_payment"] == pytest.approx(
            expected_pmt, rel=1e-4
        )

    def test_total_payment(self):
        """Total payment = annual_payment * n_years."""
        result = loan_amortization(500_000, 0.10, 10)
        assert result["total_payment"] == pytest.approx(
            result["annual_payment"] * 10, rel=1e-6
        )

    def test_total_interest(self):
        """Total interest = total_payment - principal."""
        result = loan_amortization(1_000_000, 0.08, 5)
        assert result["total_interest"] == pytest.approx(
            result["total_payment"] - 1_000_000, rel=1e-6
        )

    def test_schedule_length(self):
        """Schedule should have n_years entries."""
        result = loan_amortization(1_000_000, 0.08, 5)
        assert len(result["schedule"]) == 5

    def test_final_balance_zero(self):
        """Final balance should be approximately zero."""
        result = loan_amortization(1_000_000, 0.08, 5)
        assert result["schedule"][-1]["balance"] == pytest.approx(
            0.0, abs=0.01
        )

    def test_invalid_principal(self):
        """Non-positive principal should raise ValueError."""
        with pytest.raises(ValueError, match="principal"):
            loan_amortization(0, 0.08, 5)

    def test_invalid_n_years(self):
        """n_years < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_years"):
            loan_amortization(1_000_000, 0.08, 0)


class TestLeverageEffectIrr:
    """Tests for leverage_effect_irr."""

    def test_known_value(self):
        """Levered IRR with 60% debt at 8%, 30% tax."""
        # IRR_L = 0.15 + (0.15 - 0.08*(1-0.30)) * 0.60/0.40
        atdc = 0.08 * 0.70  # 0.056
        lev = 0.15 + (0.15 - atdc) * (0.60 / 0.40)
        result = leverage_effect_irr(0.15, 0.60, 0.08, 0.30)
        assert result == pytest.approx(lev, rel=1e-4)

    def test_no_debt(self):
        """Zero debt fraction should return unlevered IRR."""
        result = leverage_effect_irr(0.15, 0.0, 0.08, 0.30)
        assert result == pytest.approx(0.15, rel=1e-6)

    def test_leverage_amplifies(self):
        """When unlevered IRR > after-tax debt cost, leverage amplifies."""
        result = leverage_effect_irr(0.15, 0.50, 0.06, 0.30)
        assert result > 0.15

    def test_invalid_debt_fraction(self):
        """Debt fraction >= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="debt_fraction"):
            leverage_effect_irr(0.15, 1.0, 0.08, 0.30)


class TestBreakEvenMetalPrice:
    """Tests for break_even_metal_price."""

    def test_known_value(self):
        """$50M costs / (100k tonnes * 0.90 recovery)."""
        expected = 50_000_000 / (100_000 * 0.90)
        result = break_even_metal_price(50_000_000, 100_000, 0.90)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_recovery_lower_price(self):
        """Higher recovery should lower the break-even price."""
        p_low = break_even_metal_price(50_000_000, 100_000, 0.70)
        p_high = break_even_metal_price(50_000_000, 100_000, 0.95)
        assert p_high < p_low

    def test_higher_costs_higher_price(self):
        """Higher costs raise the break-even price."""
        p_low = break_even_metal_price(30_000_000, 100_000, 0.90)
        p_high = break_even_metal_price(80_000_000, 100_000, 0.90)
        assert p_high > p_low

    def test_invalid_recovery(self):
        """Recovery outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="recovery"):
            break_even_metal_price(50_000_000, 100_000, 0.0)

    def test_invalid_production(self):
        """Non-positive production should raise ValueError."""
        with pytest.raises(ValueError, match="annual_production"):
            break_even_metal_price(50_000_000, 0, 0.90)


class TestWorkingCapitalRequirement:
    """Tests for working_capital_requirement."""

    def test_known_value(self):
        """$36.5M OPEX, 45-day cycle = $4.5M."""
        result = working_capital_requirement(36_500_000, 45)
        assert result == pytest.approx(4_500_000.0, rel=1e-4)

    def test_longer_cycle_more_capital(self):
        """Longer cash cycle should require more working capital."""
        wc_short = working_capital_requirement(36_500_000, 30)
        wc_long = working_capital_requirement(36_500_000, 90)
        assert wc_long > wc_short

    def test_proportional_to_opex(self):
        """Working capital should be proportional to OPEX."""
        wc1 = working_capital_requirement(10_000_000, 45)
        wc2 = working_capital_requirement(20_000_000, 45)
        assert wc2 == pytest.approx(2 * wc1, rel=1e-6)

    def test_invalid_opex(self):
        """Non-positive OPEX should raise ValueError."""
        with pytest.raises(ValueError, match="annual_opex"):
            working_capital_requirement(0, 45)

    def test_invalid_cycle_days(self):
        """Non-positive cycle days should raise ValueError."""
        with pytest.raises(ValueError, match="cash_cycle_days"):
            working_capital_requirement(36_500_000, 0)
