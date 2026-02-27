"""Tests for minelab.geostatistics.transformations."""

import numpy as np
import pytest

from minelab.geostatistics.transformations import (
    back_transform,
    gaussian_anamorphosis,
    indicator_transform,
    lognormal_transform,
    normal_score_transform,
)


class TestNormalScoreTransform:
    """Tests for normal score transformation."""

    def test_output_mean_near_zero(self):
        """Transformed data should have mean ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 1000)
        result = normal_score_transform(data)
        assert abs(np.mean(result["transformed"])) < 0.1

    def test_output_std_near_one(self):
        """Transformed data should have std ≈ 1."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 1000)
        result = normal_score_transform(data)
        assert np.std(result["transformed"]) == pytest.approx(1.0, abs=0.1)

    def test_transform_table_shape(self):
        """Transform table should have 2 columns and n rows."""
        data = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = normal_score_transform(data)
        assert result["transform_table"].shape == (5, 2)

    def test_preserves_order(self):
        """Larger original values should get larger normal scores."""
        data = np.array([1.0, 5.0, 3.0])
        result = normal_score_transform(data)
        ns = result["transformed"]
        # index 0 (val=1) < index 2 (val=3) < index 1 (val=5)
        assert ns[0] < ns[2] < ns[1]


class TestBackTransform:
    """Tests for back-transformation."""

    def test_roundtrip(self):
        """back_transform(normal_score_transform(x)) ≈ x (sorted)."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 100)
        result = normal_score_transform(data)
        bt = back_transform(result["transformed"], result["transform_table"])
        np.testing.assert_allclose(np.sort(bt), np.sort(data), rtol=1e-4)

    def test_extrapolation_clamps(self):
        """Values beyond table range should be clamped."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normal_score_transform(data)
        bt = back_transform(np.array([-10.0, 10.0]), result["transform_table"])
        assert bt[0] == pytest.approx(1.0, abs=0.1)
        assert bt[1] == pytest.approx(5.0, abs=0.1)


class TestGaussianAnamorphosis:
    """Tests for Gaussian anamorphosis."""

    def test_coefficients_count(self):
        """Should return n_hermite coefficients."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 500)
        result = gaussian_anamorphosis(data, n_hermite=20)
        assert len(result["coefficients"]) == 20

    def test_includes_transform_table(self):
        """Should include a transform table for back-transform."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 200)
        result = gaussian_anamorphosis(data, n_hermite=10)
        assert "transform_table" in result
        assert result["transform_table"].shape[1] == 2

    def test_first_coefficient_is_mean(self):
        """First Hermite coefficient c_0 ≈ mean of data (He_0 = 1)."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 0.5, 1000)
        result = gaussian_anamorphosis(data, n_hermite=10)
        # c_0 = (1/n) * Σ z_i * He_0(y_i) / 0! = mean(z)
        assert result["coefficients"][0] == pytest.approx(np.mean(data), rel=0.05)


class TestIndicatorTransform:
    """Tests for indicator transformation."""

    def test_basic(self):
        """Values ≤ cutoff should be 1, else 0."""
        data = np.array([1, 3, 5, 7, 9])
        ind = indicator_transform(data, [4, 6])
        expected_c1 = np.array([1, 1, 0, 0, 0])
        expected_c2 = np.array([1, 1, 1, 0, 0])
        np.testing.assert_array_equal(ind[:, 0], expected_c1)
        np.testing.assert_array_equal(ind[:, 1], expected_c2)

    def test_sum_constraint(self):
        """Sum of indicators for each cutoff ≤ n_data."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 1, 100)
        cutoffs = [0.5, 1.0, 2.0, 5.0]
        ind = indicator_transform(data, cutoffs)
        for k in range(len(cutoffs)):
            assert np.sum(ind[:, k]) <= len(data)

    def test_monotonic_in_cutoffs(self):
        """Higher cutoff → more (or equal) indicators = 1."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        ind = indicator_transform(data, [2, 3, 4])
        sums = np.sum(ind, axis=0)
        assert np.all(np.diff(sums) >= 0)

    def test_shape(self):
        """Output shape should be (n, n_cutoffs)."""
        data = np.arange(10, dtype=float)
        cutoffs = [3, 5, 7]
        ind = indicator_transform(data, cutoffs)
        assert ind.shape == (10, 3)


class TestLognormalTransform:
    """Tests for lognormal transform."""

    def test_lognormal_data_detected(self):
        """Lognormal data should be detected as lognormal."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(2, 0.5, 500)
        result = lognormal_transform(data)
        assert result["is_lognormal"] is True

    def test_log_values_normal(self):
        """Log of lognormal data should be approximately normal."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(2, 0.5, 500)
        result = lognormal_transform(data)
        assert result["mean_log"] == pytest.approx(2.0, abs=0.2)
        assert result["var_log"] == pytest.approx(0.25, abs=0.1)

    def test_negative_data_raises(self):
        """Non-positive data should raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            lognormal_transform(np.array([1, 2, -1, 3]))

    def test_zero_data_raises(self):
        """Zero in data should raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            lognormal_transform(np.array([1, 0, 3]))
