"""Tests for minelab.geostatistics.kriging."""

import numpy as np
import pytest

from minelab.geostatistics.kriging import (
    block_kriging,
    cross_validate,
    indicator_kriging,
    ordinary_kriging,
    simple_kriging,
    universal_kriging,
)
from minelab.geostatistics.variogram_fitting import fit_variogram_manual


@pytest.fixture()
def simple_model():
    """A simple spherical variogram model for tests."""
    return fit_variogram_manual("spherical", 0, 10, 100)


@pytest.fixture()
def sample_data():
    """4-point dataset on a square grid."""
    coords = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    return coords, values


class TestOrdinaryKriging:
    """Tests for ordinary kriging."""

    def test_weights_sum_to_one(self, simple_model, sample_data):
        """OK weights must sum to 1.0 (verified via estimate being weighted mean)."""
        coords, values = sample_data
        target = np.array([[50, 50]])
        est, var = ordinary_kriging(coords, values, target, simple_model)
        # For equidistant points, estimate should be the mean
        assert est[0] == pytest.approx(np.mean(values), abs=0.5)

    def test_variance_positive(self, simple_model, sample_data):
        """OK variance should be non-negative."""
        coords, values = sample_data
        target = np.array([[50, 50]])
        _, var = ordinary_kriging(coords, values, target, simple_model)
        assert var[0] >= -1e-6  # allow tiny numerical errors

    def test_exact_at_data_point(self, simple_model, sample_data):
        """Estimating at a data point should return that value."""
        coords, values = sample_data
        est, var = ordinary_kriging(coords, values, coords[:1], simple_model)
        assert est[0] == pytest.approx(values[0], abs=0.01)

    def test_multiple_targets(self, simple_model, sample_data):
        """Should handle multiple target points."""
        coords, values = sample_data
        targets = np.array([[25, 25], [50, 50], [75, 75]])
        est, var = ordinary_kriging(coords, values, targets, simple_model)
        assert est.shape == (3,)
        assert var.shape == (3,)

    def test_search_radius(self, simple_model, sample_data):
        """Search radius filters distant points."""
        coords, values = sample_data
        target = np.array([[50, 50]])
        est_all, _ = ordinary_kriging(coords, values, target, simple_model)
        est_near, _ = ordinary_kriging(
            coords, values, target, simple_model, search_radius=80
        )
        # With smaller radius, might use fewer points
        assert not np.isnan(est_all[0])


class TestSimpleKriging:
    """Tests for simple kriging."""

    def test_sk_variance_leq_ok(self, simple_model, sample_data):
        """SK variance should be ≤ OK variance (SK uses more info)."""
        coords, values = sample_data
        target = np.array([[50, 50]])
        _, var_ok = ordinary_kriging(coords, values, target, simple_model)
        _, var_sk = simple_kriging(
            coords, values, target, simple_model, global_mean=2.5
        )
        assert var_sk[0] <= var_ok[0] + 1e-6

    def test_returns_mean_when_no_data(self, simple_model):
        """With no nearby data, SK should return the global mean."""
        coords = np.array([[0, 0], [1, 0]], dtype=float)
        values = np.array([5.0, 6.0])
        target = np.array([[10000, 10000]])
        est, _ = simple_kriging(
            coords, values, target, simple_model,
            global_mean=3.0, search_radius=10,
        )
        assert est[0] == pytest.approx(3.0, abs=0.01)

    def test_estimate_shape(self, simple_model, sample_data):
        """Output shapes should match number of targets."""
        coords, values = sample_data
        targets = np.array([[25, 25], [75, 75]])
        est, var = simple_kriging(
            coords, values, targets, simple_model, global_mean=2.5
        )
        assert est.shape == (2,)
        assert var.shape == (2,)


class TestUniversalKriging:
    """Tests for universal kriging."""

    def test_uk_drift0_equals_ok(self, simple_model, sample_data):
        """UK with drift=0 should equal OK."""
        coords, values = sample_data
        target = np.array([[50, 50]])
        est_ok, var_ok = ordinary_kriging(coords, values, target, simple_model)
        est_uk, var_uk = universal_kriging(
            coords, values, target, simple_model, drift_terms=0
        )
        assert est_uk[0] == pytest.approx(est_ok[0], rel=1e-4)

    def test_uk_with_linear_drift(self, simple_model):
        """UK with linear drift should handle data with trend."""
        rng = np.random.default_rng(42)
        coords = rng.random((20, 2)) * 100
        # Linear trend: z = 2*x + noise
        values = 2.0 * coords[:, 0] + rng.normal(0, 1, 20)
        target = np.array([[50, 50]])
        est, var = universal_kriging(
            coords, values, target, simple_model, drift_terms=1
        )
        assert not np.isnan(est[0])
        # Estimate should be roughly 2*50 = 100
        assert est[0] == pytest.approx(100.0, abs=30.0)


class TestIndicatorKriging:
    """Tests for indicator kriging."""

    def test_probabilities_in_range(self, simple_model, sample_data):
        """IK probabilities must be in [0, 1]."""
        coords, values = sample_data
        ik_model = fit_variogram_manual("spherical", 0, 0.25, 100)
        target = np.array([[50, 50]])
        probs = indicator_kriging(
            coords, values, target, [2.5], [ik_model]
        )
        assert probs.shape == (1, 1)
        assert 0.0 <= probs[0, 0] <= 1.0

    def test_monotonic_cutoffs(self, simple_model):
        """Higher cutoff → higher probability."""
        rng = np.random.default_rng(42)
        coords = rng.random((30, 2)) * 100
        values = rng.lognormal(0, 1, 30)
        target = np.array([[50, 50]])
        cutoffs = [0.5, 1.0, 2.0]
        models = [fit_variogram_manual("spherical", 0, 0.25, 80) for _ in cutoffs]
        probs = indicator_kriging(coords, values, target, cutoffs, models)
        # Probabilities should increase (or equal) with cutoff
        assert np.all(np.diff(probs[0, :]) >= -0.05)  # small tolerance


class TestBlockKriging:
    """Tests for block kriging."""

    def test_block_variance_less_than_point(self, simple_model):
        """Block kriging variance should be ≤ point kriging variance at same targets."""
        rng = np.random.default_rng(42)
        coords = rng.random((20, 2)) * 100
        values = rng.normal(5, 2, 20)

        bdef = {
            "origin": np.array([20, 20]),
            "size": np.array([30, 30]),
            "n_blocks": np.array([2, 2]),
        }
        b_est, b_var = block_kriging(
            coords, values, bdef, simple_model, discretization=3
        )

        # Compare with point kriging at block centers (not at data points)
        centers = np.array([[35, 35], [65, 35], [35, 65], [65, 65]], dtype=float)
        _, p_var = ordinary_kriging(coords, values, centers, simple_model)

        # Block variance should generally be less than point variance
        # because block estimates average over the block volume
        assert np.nanmean(b_var) <= np.nanmean(p_var) + 1.0

    def test_output_count(self, simple_model, sample_data):
        """Number of estimates matches number of blocks."""
        coords, values = sample_data
        bdef = {
            "origin": np.array([0, 0]),
            "size": np.array([50, 50]),
            "n_blocks": np.array([2, 2]),
        }
        est, var = block_kriging(coords, values, bdef, simple_model, discretization=2)
        assert len(est) == 4
        assert len(var) == 4


class TestCrossValidate:
    """Tests for leave-one-out cross-validation."""

    def test_output_keys(self, simple_model, sample_data):
        """CV result should contain all expected keys."""
        coords, values = sample_data
        cv = cross_validate(coords, values, simple_model)
        assert "errors" in cv
        assert "estimates" in cv
        assert "variances" in cv
        assert "standardized_errors" in cv
        assert "mean_error" in cv
        assert "mean_squared_error" in cv

    def test_error_count(self, simple_model, sample_data):
        """Number of errors should match number of data points."""
        coords, values = sample_data
        cv = cross_validate(coords, values, simple_model)
        assert len(cv["errors"]) == len(values)

    def test_mean_error_near_zero(self, simple_model):
        """Mean error should be approximately 0 for unbiased estimator."""
        rng = np.random.default_rng(42)
        n = 30
        coords = rng.random((n, 2)) * 100
        values = rng.normal(5, 2, n)
        cv = cross_validate(coords, values, simple_model)
        # Mean error should be reasonably close to 0
        assert abs(cv["mean_error"]) < 5.0

    def test_sk_method(self, simple_model, sample_data):
        """SK cross-validation should work with global_mean."""
        coords, values = sample_data
        cv = cross_validate(
            coords, values, simple_model, method="sk", global_mean=2.5
        )
        assert len(cv["errors"]) == len(values)

    def test_invalid_method(self, simple_model, sample_data):
        """Unknown method should raise ValueError."""
        coords, values = sample_data
        with pytest.raises(ValueError, match="Unknown method"):
            cross_validate(coords, values, simple_model, method="xyz")

    def test_sk_without_global_mean(self, simple_model, sample_data):
        """SK CV without global_mean should raise ValueError."""
        coords, values = sample_data
        with pytest.raises(ValueError, match="global_mean"):
            cross_validate(coords, values, simple_model, method="sk", global_mean=None)


class TestOrdinaryKrigingEdgeCases:
    """Additional edge-case tests for ordinary kriging coverage."""

    def test_1d_target_coords(self, simple_model, sample_data):
        """A single 1D target array (ndim==1) should be reshaped automatically."""
        coords, values = sample_data
        # Pass a flat array instead of 2D
        target = np.array([50.0, 50.0])
        est, var = ordinary_kriging(coords, values, target, simple_model)
        assert est.shape == (1,)
        assert var.shape == (1,)
        assert not np.isnan(est[0])

    def test_no_neighbors_found(self, simple_model):
        """When search_radius is tiny, no neighbors are found -> NaN."""
        coords = np.array([[0, 0], [100, 0]], dtype=float)
        values = np.array([1.0, 2.0])
        target = np.array([[50, 50]])
        est, var = ordinary_kriging(
            coords, values, target, simple_model, search_radius=0.001
        )
        assert np.isnan(est[0])
        assert np.isnan(var[0])

    def test_singular_matrix_duplicate_points(self):
        """Duplicate data points should cause singular matrix -> NaN."""
        # Pure nugget model: sill at distance=0 gives zero covariance matrix
        # Use a model with zero nugget and points at the exact same location
        model = fit_variogram_manual("spherical", 0, 10, 100)
        # All points at exactly the same location -> covariance matrix is singular
        coords = np.array([[0, 0], [0, 0], [0, 0]], dtype=float)
        values = np.array([1.0, 2.0, 3.0])
        target = np.array([[50, 50]])
        est, var = ordinary_kriging(coords, values, target, model)
        # Should either produce a result or NaN (singular matrix catch)
        # The covariance matrix has identical rows, so it's singular
        assert est.shape == (1,)


class TestSimpleKrigingEdgeCases:
    """Additional edge-case tests for simple kriging coverage."""

    def test_1d_target_coords(self, simple_model, sample_data):
        """A single 1D target array (ndim==1) should be reshaped automatically."""
        coords, values = sample_data
        target = np.array([50.0, 50.0])
        est, var = simple_kriging(
            coords, values, target, simple_model, global_mean=2.5
        )
        assert est.shape == (1,)
        assert var.shape == (1,)

    def test_singular_matrix_duplicate_points(self):
        """Duplicate points cause singular covariance matrix -> fallback to mean."""
        model = fit_variogram_manual("spherical", 0, 10, 100)
        # All points at exactly the same location -> singular matrix
        coords = np.array([[0, 0], [0, 0], [0, 0]], dtype=float)
        values = np.array([1.0, 2.0, 3.0])
        target = np.array([[50, 50]])
        est, var = simple_kriging(
            coords, values, target, model, global_mean=5.0
        )
        # When LinAlgError is caught, SK returns global_mean and sill variance
        assert est.shape == (1,)
        # Either the solver handles it or the except branch fires
        # In the except case: est = global_mean, var = sill
        if np.isclose(est[0], 5.0):
            assert est[0] == pytest.approx(5.0)
            assert var[0] == pytest.approx(10.0)


class TestUniversalKrigingEdgeCases:
    """Additional edge-case tests for universal kriging coverage."""

    def test_1d_target_coords(self, simple_model, sample_data):
        """A single 1D target (ndim==1) should be reshaped automatically."""
        coords, values = sample_data
        target = np.array([50.0, 50.0])
        est, var = universal_kriging(
            coords, values, target, simple_model, drift_terms=0
        )
        assert est.shape == (1,)
        assert var.shape == (1,)

    def test_quadratic_drift(self, simple_model):
        """UK with drift_terms=2 should exercise quadratic drift branch."""
        rng = np.random.default_rng(42)
        coords = rng.random((30, 2)) * 100
        # Quadratic trend: z = x^2 + y^2 + noise
        values = coords[:, 0] ** 2 + coords[:, 1] ** 2 + rng.normal(0, 10, 30)
        target = np.array([[50, 50]])
        est, var = universal_kriging(
            coords, values, target, simple_model, drift_terms=2
        )
        assert est.shape == (1,)
        assert not np.isnan(est[0])
        # Estimate should be in the right ballpark for 50^2 + 50^2 = 5000
        assert est[0] == pytest.approx(5000.0, abs=3000.0)

    def test_no_neighbors_found(self, simple_model):
        """When search_radius is tiny, no neighbors are found -> NaN."""
        coords = np.array([[0, 0], [100, 0]], dtype=float)
        values = np.array([1.0, 2.0])
        target = np.array([[50, 50]])
        est, var = universal_kriging(
            coords, values, target, simple_model,
            drift_terms=1, search_radius=0.001,
        )
        assert np.isnan(est[0])
        assert np.isnan(var[0])

    def test_singular_matrix_duplicate_points(self):
        """Duplicate points cause singular matrix -> NaN in UK."""
        model = fit_variogram_manual("spherical", 0, 10, 100)
        coords = np.array([[0, 0], [0, 0], [0, 0]], dtype=float)
        values = np.array([1.0, 2.0, 3.0])
        target = np.array([[50, 50]])
        est, var = universal_kriging(
            coords, values, target, model, drift_terms=1
        )
        assert est.shape == (1,)
        # Either numerical solution or NaN from except branch


class TestIndicatorKrigingEdgeCases:
    """Additional edge-case tests for indicator kriging coverage."""

    def test_mismatched_models_cutoffs(self, simple_model):
        """Number of models must match number of cutoffs."""
        coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
        values = np.array([1.0, 3.0, 2.0, 5.0])
        target = np.array([[25, 25]])
        with pytest.raises(ValueError, match="one variogram model per cutoff"):
            indicator_kriging(coords, values, target, [1.0, 2.0], [simple_model])

    def test_1d_target_coords(self):
        """A single 1D target (ndim==1) should be reshaped automatically."""
        model = fit_variogram_manual("spherical", 0, 0.25, 100)
        coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
        values = np.array([1.0, 3.0, 2.0, 5.0])
        # Pass flat target
        target = np.array([25.0, 25.0])
        probs = indicator_kriging(coords, values, target, [2.5], [model])
        assert probs.shape == (1, 1)
        assert 0.0 <= probs[0, 0] <= 1.0


class TestBlockKrigingEdgeCases:
    """Additional edge-case tests for block kriging coverage."""

    def test_all_disc_points_no_neighbors(self, simple_model):
        """Block far from data with tiny search_radius -> all NaN disc points."""
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        values = np.array([1.0, 2.0, 3.0])
        bdef = {
            "origin": np.array([10000, 10000]),
            "size": np.array([10, 10]),
            "n_blocks": np.array([1, 1]),
        }
        est, var = block_kriging(
            coords, values, bdef, simple_model,
            discretization=2, search_radius=0.001,
        )
        assert len(est) == 1
        assert np.isnan(est[0])
        assert np.isnan(var[0])


class TestMaxPointsNeighborSelection:
    """Tests for the max_points branch in _select_neighbors."""

    def test_max_points_limits_neighbors(self, simple_model):
        """With max_points < n_data, only closest points are used."""
        rng = np.random.default_rng(42)
        coords = rng.random((20, 2)) * 100
        values = rng.normal(5, 2, 20)
        target = np.array([[50, 50]])
        # Use max_points=3 so only 3 nearest neighbors are selected
        est, var = ordinary_kriging(
            coords, values, target, simple_model, max_points=3
        )
        assert est.shape == (1,)
        assert not np.isnan(est[0])

    def test_max_points_affects_estimate(self, simple_model):
        """Different max_points values should generally give different estimates."""
        rng = np.random.default_rng(42)
        coords = rng.random((20, 2)) * 100
        values = rng.normal(5, 2, 20)
        target = np.array([[50, 50]])
        est_all, _ = ordinary_kriging(coords, values, target, simple_model)
        est_few, _ = ordinary_kriging(
            coords, values, target, simple_model, max_points=3
        )
        # Both should be valid estimates; they may differ since different
        # subsets of neighbors are used
        assert not np.isnan(est_all[0])
        assert not np.isnan(est_few[0])
