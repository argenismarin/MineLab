"""Tests for minelab.economics.sensitivity module."""

import pytest

from minelab.economics.sensitivity import spider_plot_data, tornado_analysis


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def linear_model(price, cost):
    """Simple profit model: profit = price - cost."""
    return price - cost


def quadratic_model(x, y):
    """Quadratic: x^2 + y."""
    return x**2 + y


# -------------------------------------------------------------------------
# Tornado Analysis
# -------------------------------------------------------------------------


class TestTornadoAnalysis:
    """Tests for tornado_analysis."""

    def test_basic_linear(self):
        base = {"price": 100, "cost": 50}
        var = {"price": (80, 120), "cost": (40, 60)}
        results = tornado_analysis(base, var, linear_model)
        assert len(results) == 2

        # price swing: (120-50)-(80-50) = 70-30 = 40
        # cost swing: (100-40)-(100-60) = 60-40 = 20
        # Sorted by descending swing, so price first
        assert results[0]["param"] == "price"
        assert results[0]["swing"] == pytest.approx(40.0)
        assert results[1]["param"] == "cost"
        assert results[1]["swing"] == pytest.approx(20.0)

    def test_base_value(self):
        """Base result should be consistent across all entries."""
        base = {"price": 100, "cost": 50}
        var = {"price": (80, 120), "cost": (40, 60)}
        results = tornado_analysis(base, var, linear_model)
        for r in results:
            assert r["base"] == pytest.approx(50.0)

    def test_sorted_by_swing(self):
        """Results should be sorted descending by swing."""
        base = {"x": 10, "y": 5}
        # x^2 + y at base: 100 + 5 = 105
        # x varied 8..12: 64+5=69, 144+5=149 -> swing=80
        # y varied 3..7: 100+3=103, 100+7=107 -> swing=4
        var = {"x": (8, 12), "y": (3, 7)}
        results = tornado_analysis(base, var, quadratic_model)
        assert results[0]["param"] == "x"
        assert results[0]["swing"] > results[1]["swing"]

    def test_low_high_values(self):
        base = {"price": 100, "cost": 50}
        var = {"price": (80, 120)}
        results = tornado_analysis(base, var, linear_model)
        r = results[0]
        assert r["low"] == pytest.approx(30.0)   # 80 - 50
        assert r["high"] == pytest.approx(70.0)   # 120 - 50

    def test_missing_key_raises(self):
        base = {"price": 100}
        var = {"cost": (40, 60)}
        with pytest.raises(KeyError, match="cost"):
            tornado_analysis(base, var, linear_model)

    def test_single_param(self):
        base = {"price": 100, "cost": 50}
        var = {"price": (90, 110)}
        results = tornado_analysis(base, var, linear_model)
        assert len(results) == 1


# -------------------------------------------------------------------------
# Spider Plot Data
# -------------------------------------------------------------------------


class TestSpiderPlotData:
    """Tests for spider_plot_data."""

    def test_basic(self):
        base = {"price": 100, "cost": 50}
        data = spider_plot_data(base, ["price", "cost"], 0.20, 5, linear_model)
        assert "price" in data
        assert "cost" in data

    def test_number_of_steps(self):
        base = {"price": 100, "cost": 50}
        data = spider_plot_data(base, ["price"], 0.20, 5, linear_model)
        pct_changes, values = data["price"]
        assert len(pct_changes) == 5
        assert len(values) == 5

    def test_pct_changes_range(self):
        """Percentage changes should span from -range*100 to +range*100."""
        base = {"price": 100, "cost": 50}
        data = spider_plot_data(base, ["price"], 0.20, 5, linear_model)
        pct_changes, _ = data["price"]
        assert pct_changes[0] == pytest.approx(-20.0)
        assert pct_changes[-1] == pytest.approx(20.0)

    def test_center_value_is_base(self):
        """At 0% change the result should equal the base model value."""
        base = {"price": 100, "cost": 50}
        data = spider_plot_data(base, ["price"], 0.20, 5, linear_model)
        pct_changes, values = data["price"]
        # Middle step should be 0%
        mid = len(pct_changes) // 2
        assert pct_changes[mid] == pytest.approx(0.0)
        assert values[mid] == pytest.approx(50.0)

    def test_monotonicity(self):
        """For a linear model where profit increases with price, values should be increasing."""
        base = {"price": 100, "cost": 50}
        data = spider_plot_data(base, ["price"], 0.20, 11, linear_model)
        _, values = data["price"]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_invalid_range_raises(self):
        base = {"price": 100}
        with pytest.raises(ValueError, match="positive"):
            spider_plot_data(base, ["price"], 0, 5, linear_model)

    def test_invalid_steps_raises(self):
        base = {"price": 100}
        with pytest.raises(ValueError, match="at least 2"):
            spider_plot_data(base, ["price"], 0.20, 1, linear_model)

    def test_missing_param_raises(self):
        base = {"price": 100}
        with pytest.raises(KeyError, match="cost"):
            spider_plot_data(base, ["cost"], 0.20, 5, linear_model)
