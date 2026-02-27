"""Tests for minelab.economics.taxation."""

import pytest

from minelab.economics.taxation import (
    after_tax_cashflow,
    capital_recovery_factor,
    income_tax_shield,
    real_to_nominal_cashflow,
    royalty_cost,
)


class TestRoyaltyCost:
    """Tests for royalty_cost."""

    def test_known_value(self):
        """5% royalty on $10M revenue = $500k."""
        result = royalty_cost(10_000_000, 0.05)
        assert result == pytest.approx(500_000.0)

    def test_zero_rate(self):
        """Zero royalty rate gives zero royalty."""
        assert royalty_cost(10_000_000, 0.0) == pytest.approx(0.0)

    def test_zero_revenue(self):
        """Zero revenue gives zero royalty."""
        assert royalty_cost(0.0, 0.05) == pytest.approx(0.0)

    def test_proportional(self):
        """Royalty should be proportional to revenue."""
        r1 = royalty_cost(5_000_000, 0.03)
        r2 = royalty_cost(10_000_000, 0.03)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)

    def test_invalid_rate(self):
        """Royalty rate > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="royalty_rate"):
            royalty_cost(10_000_000, 1.5)


class TestIncomeTaxShield:
    """Tests for income_tax_shield."""

    def test_known_value(self):
        """$2M depreciation at 30% tax = $600k shield."""
        result = income_tax_shield(2_000_000, 0.30)
        assert result == pytest.approx(600_000.0)

    def test_zero_depreciation(self):
        """Zero depreciation gives zero shield."""
        assert income_tax_shield(0.0, 0.30) == pytest.approx(0.0)

    def test_zero_tax_rate(self):
        """Zero tax rate gives zero shield."""
        assert income_tax_shield(2_000_000, 0.0) == pytest.approx(0.0)

    def test_invalid_tax_rate(self):
        """Tax rate > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="tax_rate"):
            income_tax_shield(1_000_000, 1.5)


class TestAfterTaxCashflow:
    """Tests for after_tax_cashflow."""

    def test_known_value(self):
        """EBITDA=$5M, D=$1M, t=30%, R=$500k."""
        result = after_tax_cashflow(5_000_000, 1_000_000, 0.30, 500_000)
        # taxable = 5M - 1M - 0.5M = 3.5M
        assert result["taxable_income"] == pytest.approx(3_500_000)
        # tax = 3.5M * 0.30 = 1.05M
        assert result["tax"] == pytest.approx(1_050_000)
        # AT CF = 5M - 1.05M - 0.5M = 3.45M
        assert result["after_tax_cashflow"] == pytest.approx(3_450_000)

    def test_negative_taxable_income_no_tax(self):
        """Negative taxable income should result in zero tax."""
        result = after_tax_cashflow(500_000, 1_000_000, 0.30, 200_000)
        assert result["taxable_income"] < 0
        assert result["tax"] == pytest.approx(0.0)

    def test_effective_tax_rate(self):
        """Effective rate = tax / EBITDA."""
        result = after_tax_cashflow(5_000_000, 1_000_000, 0.30, 500_000)
        expected_eff = 1_050_000.0 / 5_000_000.0
        assert result["effective_tax_rate"] == pytest.approx(
            expected_eff, rel=1e-4
        )

    def test_invalid_depreciation(self):
        """Negative depreciation should raise ValueError."""
        with pytest.raises(ValueError, match="depreciation"):
            after_tax_cashflow(5_000_000, -100, 0.30, 0)


class TestRealToNominalCashflow:
    """Tests for real_to_nominal_cashflow."""

    def test_known_value(self):
        """$1M real at 3% inflation for 5 years."""
        expected = 1_000_000 * (1.03 ** 5)
        result = real_to_nominal_cashflow(1_000_000, 0.03, 5)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_year_zero(self):
        """Year 0 should return the original cash flow."""
        result = real_to_nominal_cashflow(1_000_000, 0.05, 0)
        assert result == pytest.approx(1_000_000.0)

    def test_zero_inflation(self):
        """Zero inflation should return the original cash flow."""
        result = real_to_nominal_cashflow(1_000_000, 0.0, 10)
        assert result == pytest.approx(1_000_000.0)

    def test_negative_cashflow(self):
        """Negative cash flows should also inflate correctly."""
        result = real_to_nominal_cashflow(-500_000, 0.03, 3)
        expected = -500_000 * (1.03 ** 3)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_invalid_inflation(self):
        """Negative inflation rate should raise ValueError."""
        with pytest.raises(ValueError, match="inflation_rate"):
            real_to_nominal_cashflow(1_000_000, -0.01, 5)


class TestCapitalRecoveryFactor:
    """Tests for capital_recovery_factor."""

    def test_known_value(self):
        """CRF at 10% for 10 periods."""
        r = 0.10
        n = 10
        factor = (1 + r) ** n
        expected = r * factor / (factor - 1)
        result = capital_recovery_factor(0.10, 10)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_period(self):
        """CRF for 1 period should be 1 + rate."""
        result = capital_recovery_factor(0.10, 1)
        assert result == pytest.approx(1.10, rel=1e-6)

    def test_high_rate(self):
        """CRF should increase with rate."""
        crf_low = capital_recovery_factor(0.05, 10)
        crf_high = capital_recovery_factor(0.20, 10)
        assert crf_high > crf_low

    def test_invalid_rate(self):
        """Non-positive rate should raise ValueError."""
        with pytest.raises(ValueError, match="rate"):
            capital_recovery_factor(0.0, 10)

    def test_invalid_periods(self):
        """Non-positive periods should raise ValueError."""
        with pytest.raises(ValueError, match="n_periods"):
            capital_recovery_factor(0.10, 0)
