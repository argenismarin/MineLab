"""Tests for minelab.mineral_processing.mass_balance."""

import numpy as np
import pytest

from minelab.mineral_processing.mass_balance import (
    check_closure,
    multi_element_balance,
    reconcile_balance,
    three_product,
    two_product,
)


class TestTwoProduct:
    """Tests for two-product mass balance."""

    def test_known_value(self):
        """f=2%, c=20%, t=0.5% → C/F ≈ 0.0769."""
        result = two_product(2.0, 20.0, 0.5)
        assert result["concentrate_ratio"] == pytest.approx(0.0769, rel=0.01)

    def test_mass_conservation(self):
        """C/F + T/F = 1."""
        result = two_product(2.0, 20.0, 0.5)
        assert (result["concentrate_ratio"] + result["tailings_ratio"]) == pytest.approx(1.0)

    def test_recovery_range(self):
        """Recovery should be 0-1."""
        result = two_product(2.0, 20.0, 0.5)
        assert 0 < result["recovery"] < 1

    def test_equal_grades_raises(self):
        """c = t should raise."""
        with pytest.raises(ValueError):
            two_product(2.0, 5.0, 5.0)


class TestThreeProduct:
    """Tests for three-product mass balance."""

    def test_sum_to_one(self):
        """Ratios should sum to 1."""
        result = three_product(10, 5, 40, 2, 5, 30, 2, 1)
        total = result["c1_ratio"] + result["c2_ratio"] + result["t_ratio"]
        assert total == pytest.approx(1.0, rel=0.01)

    def test_positive_ratios(self):
        """Ratios should be reasonable."""
        result = three_product(10, 5, 40, 2, 5, 30, 2, 1)
        assert result["c1_ratio"] > 0
        assert result["t_ratio"] > 0


class TestMultiElementBalance:
    """Tests for multi-element balance."""

    def test_balanced(self):
        """Perfect balance → error = 0."""
        feed = {"Cu": 2.0, "Fe": 30.0}
        products = [{"Cu": 20.0, "Fe": 10.0}, {"Cu": 0.0, "Fe": 32.22}]
        ratios = np.array([0.1, 0.9])
        result = multi_element_balance(feed, products, ratios)
        assert result["balance_errors"]["Cu"] == pytest.approx(0.0, abs=0.01)

    def test_unbalanced(self):
        """Bad data → not balanced."""
        feed = {"Cu": 2.0}
        products = [{"Cu": 20.0}, {"Cu": 5.0}]
        ratios = np.array([0.5, 0.5])
        result = multi_element_balance(feed, products, ratios)
        assert not result["balanced"]


class TestReconcileBalance:
    """Tests for mass balance reconciliation."""

    def test_sums_to_one(self):
        """Adjusted fractions should sum to 1.0."""
        result = reconcile_balance(np.array([0.3, 0.68, 0.05]))
        assert float(result["adjusted"].sum()) == pytest.approx(1.0)

    def test_already_closed(self):
        """If already sums to 1 → closed = True."""
        result = reconcile_balance(np.array([0.3, 0.5, 0.2]))
        assert result["closed"]


class TestCheckClosure:
    """Tests for mass balance closure check."""

    def test_closed(self):
        """1030 / 1000 = 3% → closed within 5%."""
        result = check_closure(1000, [300, 680, 50], tolerance=0.05)
        assert result["closed"]

    def test_not_closed(self):
        """Large error → not closed."""
        result = check_closure(1000, [300, 500, 50])
        assert not result["closed"]

    def test_total_products(self):
        """Total should sum products."""
        result = check_closure(1000, [300, 400, 300])
        assert result["total_products"] == 1000
