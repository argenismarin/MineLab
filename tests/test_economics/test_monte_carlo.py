"""Tests for minelab.economics.monte_carlo module."""

import numpy as np
import pytest

from minelab.economics.monte_carlo import (
    confidence_intervals,
    mc_npv,
    run_monte_carlo,
    triangular_sample,
)


# -------------------------------------------------------------------------
# Triangular Sampling
# -------------------------------------------------------------------------


class TestTriangularSample:
    """Tests for the triangular_sample helper."""

    def test_bounds(self):
        """All samples should lie within [low, high]."""
        samples = triangular_sample(1, 3, 5, 10_000, rng=np.random.default_rng(0))
        assert samples.min() >= 1.0
        assert samples.max() <= 5.0

    def test_shape(self):
        samples = triangular_sample(0, 1, 2, 500, rng=np.random.default_rng(0))
        assert samples.shape == (500,)

    def test_mean_near_expected(self):
        """Mean of triangular(1,3,5) = (1+3+5)/3 = 3."""
        samples = triangular_sample(1, 3, 5, 100_000, rng=np.random.default_rng(42))
        assert samples.mean() == pytest.approx(3.0, abs=0.05)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="low <= mode <= high"):
            triangular_sample(5, 3, 1, 100)

    def test_equal_low_high_raises(self):
        with pytest.raises(ValueError, match="differ"):
            triangular_sample(3, 3, 3, 100)

    def test_none_rng(self):
        """Passing rng=None should still work."""
        samples = triangular_sample(0, 1, 2, 10)
        assert len(samples) == 10


# -------------------------------------------------------------------------
# Generic Monte Carlo Engine
# -------------------------------------------------------------------------


class TestRunMonteCarlo:
    """Tests for the generic run_monte_carlo function."""

    def test_simple_model(self):
        """Profit = price - cost with known distributions."""

        def profit(price, cost):
            return price - cost

        dists = {
            "price": ("uniform", (80, 120)),
            "cost": ("fixed", (50,)),
        }
        results = run_monte_carlo(profit, dists, 10_000, rng=np.random.default_rng(42))
        assert results.shape == (10_000,)
        # Mean profit ~ (80+120)/2 - 50 = 50
        assert results.mean() == pytest.approx(50.0, abs=2.0)

    def test_all_fixed(self):
        """With all fixed inputs, every result should be identical."""

        def add(a, b):
            return a + b

        dists = {"a": ("fixed", (10,)), "b": ("fixed", (20,))}
        results = run_monte_carlo(add, dists, 100, rng=np.random.default_rng(0))
        assert np.all(results == 30.0)

    def test_normal_distribution(self):
        """Test with normal distribution."""

        def identity(x):
            return x

        dists = {"x": ("normal", (100, 10))}
        results = run_monte_carlo(identity, dists, 50_000, rng=np.random.default_rng(7))
        assert results.mean() == pytest.approx(100.0, abs=1.0)
        assert results.std() == pytest.approx(10.0, abs=1.0)

    def test_unsupported_dist_raises(self):
        def f(x):
            return x

        dists = {"x": ("beta_unknown", (1, 2))}
        with pytest.raises(ValueError, match="Unsupported"):
            run_monte_carlo(f, dists, 10, rng=np.random.default_rng(0))

    def test_zero_iterations_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            run_monte_carlo(lambda: 0, {}, 0)


# -------------------------------------------------------------------------
# Monte Carlo NPV
# -------------------------------------------------------------------------


class TestMcNPV:
    """Tests for the Monte Carlo NPV simulation."""

    def test_deterministic(self):
        """With all fixed distributions, MC NPV should match analytic NPV."""
        from minelab.economics.cashflow import npv as analytic_npv

        cfs = [-1000, 300, 420, 680]
        dists = [("fixed", (cf,)) for cf in cfs]
        results = mc_npv(0.10, dists, 1000, rng=np.random.default_rng(0))
        expected = analytic_npv(0.10, cfs)
        assert np.all(np.isclose(results, expected, atol=1e-6))

    def test_shape(self):
        dists = [
            ("fixed", (-1000,)),
            ("triangular", (200, 300, 400)),
            ("triangular", (350, 420, 500)),
        ]
        results = mc_npv(0.10, dists, 5000, rng=np.random.default_rng(42))
        assert results.shape == (5000,)

    def test_spread(self):
        """Stochastic inputs should produce a spread of NPV values."""
        dists = [
            ("fixed", (-1000,)),
            ("uniform", (100, 500)),
            ("uniform", (100, 500)),
        ]
        results = mc_npv(0.10, dists, 5000, rng=np.random.default_rng(42))
        assert results.std() > 0

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="greater than -1"):
            mc_npv(-1.0, [("fixed", (100,))], 100)


# -------------------------------------------------------------------------
# Confidence Intervals
# -------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Tests for percentile-based confidence intervals."""

    def test_known_values(self):
        """P50 of 1..100 should be 50.5."""
        ci = confidence_intervals(np.arange(1, 101))
        assert ci["P50"] == pytest.approx(50.5)

    def test_keys(self):
        ci = confidence_intervals(np.arange(100), levels=(10, 50, 90))
        assert set(ci.keys()) == {"P10", "P50", "P90"}

    def test_ordering(self):
        """P10 <= P50 <= P90."""
        data = np.random.default_rng(0).normal(0, 1, 10_000)
        ci = confidence_intervals(data)
        assert ci["P10"] <= ci["P50"] <= ci["P90"]

    def test_custom_levels(self):
        ci = confidence_intervals(np.arange(1000), levels=(5, 25, 75, 95))
        assert "P5" in ci
        assert "P95" in ci
        assert ci["P5"] < ci["P95"]


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestSampleDistributionLognormal:
    """Test lognormal distribution sampling."""

    def test_lognormal_distribution(self):
        """Lognormal distribution should produce positive values."""

        def identity(x):
            return x

        dists = {"x": ("lognormal", (0, 0.5))}
        results = run_monte_carlo(identity, dists, 10_000, rng=np.random.default_rng(42))
        assert np.all(results > 0)
        # Mean of lognormal(0, 0.5) = exp(0 + 0.5^2/2) = exp(0.125) ~ 1.133
        assert results.mean() == pytest.approx(1.133, abs=0.1)


class TestTriangularSampleNValidation:
    """Additional validation for triangular_sample."""

    def test_zero_n_raises(self):
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            triangular_sample(1, 3, 5, 0)


class TestRunMonteCarloRngNone:
    """Test run_monte_carlo with rng=None."""

    def test_none_rng(self):
        """Passing rng=None should still work (uses default generator)."""

        def add(a, b):
            return a + b

        dists = {"a": ("fixed", (10,)), "b": ("fixed", (20,))}
        results = run_monte_carlo(add, dists, 10, rng=None)
        assert np.all(results == 30.0)


class TestMcNPVAdditional:
    """Additional tests for mc_npv."""

    def test_zero_iterations_raises(self):
        """n_iterations < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            mc_npv(0.10, [("fixed", (100,))], 0)

    def test_none_rng(self):
        """Passing rng=None should still work."""
        dists = [("fixed", (-1000,)), ("fixed", (500,)), ("fixed", (600,))]
        results = mc_npv(0.10, dists, 10, rng=None)
        assert results.shape == (10,)
