"""Tests for minelab.geostatistics.simulation."""

import numpy as np
import pytest

from minelab.geostatistics.simulation import (
    back_transform_simulation,
    sequential_gaussian_simulation,
    sequential_indicator_simulation,
    simulation_statistics,
)
from minelab.geostatistics.transformations import normal_score_transform
from minelab.geostatistics.variogram_fitting import fit_variogram_manual


@pytest.fixture()
def sgs_setup():
    """Common setup for SGS tests."""
    model = fit_variogram_manual("spherical", 0, 1, 50)
    rng = np.random.default_rng(42)
    coords = rng.random((15, 2)) * 100
    values = rng.lognormal(0, 0.5, 15)
    grid = np.column_stack([
        np.repeat(np.arange(10, 100, 20), 5),
        np.tile(np.arange(10, 100, 20), 5),
    ]).astype(float)
    return coords, values, grid, model


class TestSequentialGaussianSimulation:
    """Tests for SGS."""

    def test_output_shape(self, sgs_setup):
        """Output shape should be (n_realizations, m)."""
        coords, values, grid, model = sgs_setup
        sims = sequential_gaussian_simulation(
            coords, values, grid, model, n_realizations=3, seed=42
        )
        assert sims.shape == (3, len(grid))

    def test_reproducibility(self, sgs_setup):
        """Same seed should give same results."""
        coords, values, grid, model = sgs_setup
        s1 = sequential_gaussian_simulation(
            coords, values, grid, model, n_realizations=1, seed=99
        )
        s2 = sequential_gaussian_simulation(
            coords, values, grid, model, n_realizations=1, seed=99
        )
        np.testing.assert_array_equal(s1, s2)

    def test_different_realizations(self, sgs_setup):
        """Different realizations should differ."""
        coords, values, grid, model = sgs_setup
        sims = sequential_gaussian_simulation(
            coords, values, grid, model, n_realizations=2, seed=42
        )
        assert not np.allclose(sims[0], sims[1])

    def test_values_positive_for_lognormal(self, sgs_setup):
        """Back-transformed lognormal data should be mostly positive."""
        coords, values, grid, model = sgs_setup
        sims = sequential_gaussian_simulation(
            coords, values, grid, model, n_realizations=1, seed=42
        )
        # Most values should be positive since input was lognormal
        assert np.mean(sims > 0) > 0.5


class TestSequentialIndicatorSimulation:
    """Tests for SIS."""

    def test_output_shape(self):
        """Output shape should be (n_realizations, m)."""
        model = fit_variogram_manual("spherical", 0, 0.25, 50)
        coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
        indicators = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
        grid = np.array([[25, 25]], dtype=float)
        sims = sequential_indicator_simulation(
            coords, indicators, grid, [model, model], [1.0, 2.0],
            n_realizations=5, seed=42,
        )
        assert sims.shape == (5, 1)

    def test_category_range(self):
        """Categories should be in [0, n_cutoffs]."""
        model = fit_variogram_manual("spherical", 0, 0.25, 50)
        coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
        indicators = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
        grid = np.array([[25, 25], [30, 30]], dtype=float)
        sims = sequential_indicator_simulation(
            coords, indicators, grid, [model, model], [1.0, 2.0],
            n_realizations=10, seed=42,
        )
        assert np.all(sims >= 0)
        assert np.all(sims <= 2)


class TestBackTransformSimulation:
    """Tests for back_transform_simulation."""

    def test_roundtrip(self):
        """Back-transforming normal scores should recover original data."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 100)
        ns = normal_score_transform(data)
        bt = back_transform_simulation(
            ns["transformed"], data, ns["transform_table"]
        )
        np.testing.assert_allclose(np.sort(bt), np.sort(data), rtol=1e-4)

    def test_2d_input(self):
        """Should handle 2D input (n_realizations, m)."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 100)
        ns = normal_score_transform(data)
        sim_2d = np.vstack([ns["transformed"], ns["transformed"]])
        bt = back_transform_simulation(sim_2d, data, ns["transform_table"])
        assert bt.shape == (2, 100)


class TestSimulationStatistics:
    """Tests for simulation_statistics."""

    def test_etype_shape(self):
        """E-type should have same length as grid."""
        rng = np.random.default_rng(42)
        reals = rng.normal(5, 2, size=(100, 20))
        stats = simulation_statistics(reals)
        assert stats["e_type"].shape == (20,)

    def test_etype_near_mean(self):
        """E-type from many realizations ≈ population mean."""
        rng = np.random.default_rng(42)
        reals = rng.normal(5, 2, size=(1000, 10))
        stats = simulation_statistics(reals)
        np.testing.assert_allclose(stats["e_type"], 5.0, atol=0.3)

    def test_percentile_ordering(self):
        """P10 ≤ P50 ≤ P90."""
        rng = np.random.default_rng(42)
        reals = rng.normal(5, 2, size=(100, 10))
        stats = simulation_statistics(reals)
        assert np.all(stats["p10"] <= stats["p50"])
        assert np.all(stats["p50"] <= stats["p90"])

    def test_variance_positive(self):
        """Conditional variance should be positive."""
        rng = np.random.default_rng(42)
        reals = rng.normal(5, 2, size=(100, 10))
        stats = simulation_statistics(reals)
        assert np.all(stats["variance"] > 0)
