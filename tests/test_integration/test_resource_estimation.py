"""Integration test for resource estimation workflow.

Tests the full pipeline: synthetic drillhole data -> experimental variogram ->
variogram fitting -> ordinary kriging -> classification by kriging variance ->
resource statement generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from minelab.geostatistics import (
    BlockModel,
    experimental_variogram,
    fit_variogram_wls,
    ordinary_kriging,
)
from minelab.resource_classification import (
    classify_by_kriging_variance,
    resource_statement,
)


def _synthetic_deposit(n_samples: int = 30, seed: int = 99) -> tuple[
    np.ndarray, np.ndarray
]:
    """Generate synthetic sample data resembling a porphyry Cu deposit.

    Returns coordinates (n, 2) and grade values (n,).
    """
    rng = np.random.default_rng(seed)
    # Random sample locations within a 200 x 200 m area
    coords = rng.uniform(0, 200, size=(n_samples, 2))
    # Grade increases toward centre (100, 100) with random noise
    dist_from_centre = np.sqrt(
        (coords[:, 0] - 100) ** 2 + (coords[:, 1] - 100) ** 2
    )
    grades = np.maximum(
        0.0,
        2.0 - dist_from_centre / 80.0 + rng.normal(0, 0.3, n_samples),
    )
    return coords, grades


@pytest.fixture()
def deposit_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic deposit sample data."""
    return _synthetic_deposit()


@pytest.fixture()
def fitted_model(deposit_data):
    """Fitted variogram model from synthetic data."""
    coords, values = deposit_data
    ev = experimental_variogram(coords, values, n_lags=8)
    model = fit_variogram_wls(
        ev["lags"], ev["semivariance"], "spherical", ev["n_pairs"]
    )
    return model


@pytest.fixture()
def kriging_results(deposit_data, fitted_model):
    """Kriging estimates and variances on a block grid."""
    coords, values = deposit_data
    # Small block model: 8 x 8 blocks of 25 m
    bm = BlockModel([0, 0, 0], [25, 25, 1], [8, 8, 1])
    centers = bm.block_centers()
    target_2d = centers[:, :2]
    estimates, variances = ordinary_kriging(
        coords, values, target_2d, fitted_model
    )
    return bm, estimates, variances


class TestVariogramFitting:
    """Verify the variogram fitting stage."""

    def test_experimental_variogram_produces_valid_lags(self, deposit_data):
        """Experimental variogram should produce monotonically increasing lags."""
        coords, values = deposit_data
        ev = experimental_variogram(coords, values, n_lags=8)
        lags = ev["lags"]
        # Non-NaN lags should be increasing
        valid = ~np.isnan(lags)
        valid_lags = lags[valid]
        assert len(valid_lags) >= 4
        for i in range(len(valid_lags) - 1):
            assert valid_lags[i + 1] > valid_lags[i]

    def test_fitted_model_parameters_physical(self, fitted_model):
        """Fitted model should have physically meaningful parameters."""
        assert fitted_model.nugget >= 0
        assert fitted_model.sill > fitted_model.nugget
        assert fitted_model.range_a > 0
        assert fitted_model.rmse >= 0

    def test_model_prediction_at_zero_is_zero(self, fitted_model):
        """Variogram at h=0 should be 0 (by definition, gamma(0)=0)."""
        gamma_0 = fitted_model.predict(0.0)
        assert gamma_0 == pytest.approx(0.0, abs=1e-10)

    def test_model_prediction_at_large_h_near_sill(self, fitted_model):
        """Variogram at h >> range should approach the sill."""
        gamma_far = fitted_model.predict(fitted_model.range_a * 10)
        assert gamma_far == pytest.approx(fitted_model.sill, rel=0.05)


class TestKrigingEstimation:
    """Verify kriging estimation quality on the block model."""

    def test_estimates_within_data_range(self, deposit_data, kriging_results):
        """Kriging estimates should not exceed the data range significantly."""
        _, values = deposit_data
        _, estimates, _ = kriging_results

        # Allow a small buffer beyond data range
        data_min, data_max = values.min(), values.max()
        margin = (data_max - data_min) * 0.3
        valid = ~np.isnan(estimates)
        assert np.all(estimates[valid] >= data_min - margin)
        assert np.all(estimates[valid] <= data_max + margin)

    def test_variances_non_negative(self, kriging_results):
        """Kriging variances should be non-negative (or very close to zero)."""
        _, _, variances = kriging_results
        valid = ~np.isnan(variances)
        assert np.all(variances[valid] >= -1e-6)

    def test_variance_lower_near_data(self, deposit_data, fitted_model):
        """Variance at a data point should be lower than at a far point."""
        coords, values = deposit_data
        near_target = coords[:1]
        far_target = np.array([[999.0, 999.0]])

        _, var_near = ordinary_kriging(coords, values, near_target, fitted_model)
        _, var_far = ordinary_kriging(coords, values, far_target, fitted_model)

        assert var_near[0] < var_far[0]


class TestClassification:
    """Verify resource classification by kriging variance."""

    def test_all_categories_present(self, kriging_results):
        """With appropriate thresholds, all three categories should appear."""
        _, estimates, variances = kriging_results
        valid = ~np.isnan(variances)
        kv = variances[valid]

        # Choose thresholds that should split the data
        p33 = np.percentile(kv, 33)
        p66 = np.percentile(kv, 66)
        thresholds = {"measured": p33, "indicated": p66}

        classification = classify_by_kriging_variance(kv, thresholds)
        unique_cats = set(classification.tolist())
        assert 1 in unique_cats, "Measured category missing"
        assert 2 in unique_cats, "Indicated category missing"
        assert 3 in unique_cats, "Inferred category missing"

    def test_measured_has_lowest_variance(self, kriging_results):
        """Blocks classified as Measured should have lower variance than Inferred."""
        _, _, variances = kriging_results
        valid = ~np.isnan(variances)
        kv = variances[valid]

        p33 = np.percentile(kv, 33)
        p66 = np.percentile(kv, 66)
        thresholds = {"measured": p33, "indicated": p66}

        classification = classify_by_kriging_variance(kv, thresholds)

        mean_var_measured = kv[classification == 1].mean()
        mean_var_inferred = kv[classification == 3].mean()
        assert mean_var_measured < mean_var_inferred

    def test_classification_counts_sum_to_total(self, kriging_results):
        """Total classified blocks should equal total valid blocks."""
        _, _, variances = kriging_results
        valid = ~np.isnan(variances)
        kv = variances[valid]
        n_total = len(kv)

        thresholds = {"measured": 0.3, "indicated": 0.7}
        classification = classify_by_kriging_variance(kv, thresholds)

        n_m = np.sum(classification == 1)
        n_i = np.sum(classification == 2)
        n_inf = np.sum(classification == 3)
        assert n_m + n_i + n_inf == n_total


class TestResourceStatement:
    """Verify resource statement from classified block model."""

    def test_statement_categories_present(self, kriging_results):
        """Resource statement should contain all three category keys."""
        bm, estimates, variances = kriging_results
        valid = ~np.isnan(variances) & ~np.isnan(estimates)
        kv = variances[valid]
        grades = np.maximum(estimates[valid], 0.0)

        p33 = np.percentile(kv, 33)
        p66 = np.percentile(kv, 66)
        classification = classify_by_kriging_variance(
            kv, {"measured": p33, "indicated": p66}
        )

        block_vol = float(np.prod(bm.block_size))
        density = 2.7
        tonnages = np.full(len(grades), density * block_vol)

        stmt = resource_statement(
            tonnages, grades, classification, cutoff=0.0
        )

        assert "measured" in stmt
        assert "indicated" in stmt
        assert "inferred" in stmt

    def test_tonnages_sum_correctly(self, kriging_results):
        """Sum of category tonnages should equal the total above cut-off."""
        bm, estimates, variances = kriging_results
        valid = ~np.isnan(variances) & ~np.isnan(estimates)
        kv = variances[valid]
        grades = np.maximum(estimates[valid], 0.0)

        p33 = np.percentile(kv, 33)
        p66 = np.percentile(kv, 66)
        classification = classify_by_kriging_variance(
            kv, {"measured": p33, "indicated": p66}
        )

        block_vol = float(np.prod(bm.block_size))
        density = 2.7
        tonnages = np.full(len(grades), density * block_vol)

        cutoff = 0.5
        stmt = resource_statement(tonnages, grades, classification, cutoff=cutoff)

        total_above = float(tonnages[grades >= cutoff].sum())
        stmt_total = (
            stmt["measured"]["tonnes"]
            + stmt["indicated"]["tonnes"]
            + stmt["inferred"]["tonnes"]
        )
        assert stmt_total == pytest.approx(total_above, rel=1e-8)

    def test_metal_equals_tonnage_times_grade(self, kriging_results):
        """Contained metal should equal tonnes * mean grade for each category."""
        bm, estimates, variances = kriging_results
        valid = ~np.isnan(variances) & ~np.isnan(estimates)
        kv = variances[valid]
        grades = np.maximum(estimates[valid], 0.0)

        p33 = np.percentile(kv, 33)
        p66 = np.percentile(kv, 66)
        classification = classify_by_kriging_variance(
            kv, {"measured": p33, "indicated": p66}
        )

        block_vol = float(np.prod(bm.block_size))
        density = 2.7
        tonnages = np.full(len(grades), density * block_vol)

        stmt = resource_statement(tonnages, grades, classification, cutoff=0.0)

        for cat in ["measured", "indicated", "inferred"]:
            t = stmt[cat]["tonnes"]
            g = stmt[cat]["grade"]
            m = stmt[cat]["metal"]
            if t > 0:
                assert m == pytest.approx(t * g, rel=1e-6)
            else:
                assert m == pytest.approx(0.0, abs=1e-10)

    def test_higher_cutoff_reduces_tonnage(self, kriging_results):
        """Raising the cut-off grade should reduce total tonnage."""
        bm, estimates, variances = kriging_results
        valid = ~np.isnan(variances) & ~np.isnan(estimates)
        kv = variances[valid]
        grades = np.maximum(estimates[valid], 0.0)

        classification = classify_by_kriging_variance(
            kv, {"measured": 0.3, "indicated": 0.7}
        )

        block_vol = float(np.prod(bm.block_size))
        tonnages = np.full(len(grades), 2.7 * block_vol)

        stmt_low = resource_statement(
            tonnages, grades, classification, cutoff=0.0
        )
        stmt_high = resource_statement(
            tonnages, grades, classification, cutoff=1.0
        )

        total_low = sum(stmt_low[c]["tonnes"] for c in ["measured", "indicated", "inferred"])
        total_high = sum(
            stmt_high[c]["tonnes"] for c in ["measured", "indicated", "inferred"]
        )
        assert total_high <= total_low
