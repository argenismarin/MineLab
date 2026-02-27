"""Tests for minelab.utilities.statistics."""

import numpy as np
import pandas as pd
import pytest

from minelab.utilities.statistics import (
    capping_analysis,
    contact_analysis,
    descriptive_stats,
    log_stats,
    probability_plot,
)


class TestDescriptiveStats:
    def test_known_dataset(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        s = descriptive_stats(data)
        assert s["count"] == 8.0
        assert s["mean"] == 5.0
        # sample variance (ddof=1): sum((xi-5)^2)/7 = 32/7
        assert pytest.approx(s["var"], rel=1e-6) == 32.0 / 7.0
        assert pytest.approx(s["std"], rel=1e-6) == (32.0 / 7.0) ** 0.5
        assert s["min"] == 2.0
        assert s["max"] == 9.0

    def test_single_value(self):
        s = descriptive_stats([42.0])
        assert s["count"] == 1.0
        assert s["mean"] == 42.0

    def test_cv(self):
        data = [10, 10, 10]
        s = descriptive_stats(data)
        assert s["cv"] == 0.0

    def test_percentiles(self):
        data = list(range(1, 101))  # 1..100
        s = descriptive_stats(data)
        assert s["p50"] == pytest.approx(50.5)
        assert s["p25"] == pytest.approx(25.75)
        assert s["p75"] == pytest.approx(75.25)

    def test_keys_present(self):
        s = descriptive_stats([1, 2, 3])
        expected_keys = {
            "count", "mean", "var", "std", "cv",
            "skew", "kurtosis", "min", "max",
            "p25", "p50", "p75",
        }
        assert set(s.keys()) == expected_keys

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            descriptive_stats([])


class TestLogStats:
    def test_known_values(self):
        data = [1, np.e, np.e**2]
        s = log_stats(data)
        # ln values: [0, 1, 2], mean = 1
        assert pytest.approx(s["mean"], rel=1e-6) == 1.0

    def test_drops_non_positive(self):
        data = [-5, 0, 1, 10, 100]
        s = log_stats(data)
        assert s["n_dropped"] == 2.0
        assert s["count"] == 3.0

    def test_all_negative_raises(self):
        with pytest.raises(ValueError, match="No positive values"):
            log_stats([-1, -2, -3])


class TestContactAnalysis:
    def test_basic(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 100)
        coords = np.linspace(0, 100, 100)
        df = contact_analysis(data, coords, 0, 10.0, 10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "lag_center" in df.columns
        assert "mean" in df.columns
        assert "count" in df.columns

    def test_lag_bins_cover_range(self):
        data = np.ones(50)
        coords = np.arange(50, dtype=float)
        df = contact_analysis(data, coords, 0, 10.0, 5)
        # All bins should have roughly 10 samples
        assert df["count"].sum() == 50

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            contact_analysis([1, 2, 3], [1, 2], 0, 1.0, 1)


class TestCappingAnalysis:
    def test_basic(self):
        np.random.seed(0)
        data = np.random.lognormal(0, 1, 1000)
        df = capping_analysis(data, [90, 95, 99])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "threshold" in df.columns
        assert "capped_mean" in df.columns
        assert "n_capped" in df.columns
        assert "pct_metal_removed" in df.columns

    def test_higher_percentile_higher_threshold(self):
        np.random.seed(0)
        data = np.random.lognormal(0, 1, 500)
        df = capping_analysis(data, [90, 95, 99])
        thresholds = df["threshold"].tolist()
        assert thresholds[0] <= thresholds[1] <= thresholds[2]

    def test_capping_reduces_mean(self):
        np.random.seed(0)
        data = np.random.lognormal(0, 2, 500)
        df = capping_analysis(data, [90])
        assert df.iloc[0]["capped_mean"] <= np.mean(data)


class TestProbabilityPlot:
    def test_output_shape(self):
        data = np.random.normal(0, 1, 100)
        sd, tq = probability_plot(data)
        assert sd.shape == (100,)
        assert tq.shape == (100,)

    def test_sorted_data(self):
        data = [5, 3, 1, 4, 2]
        sd, _ = probability_plot(data)
        np.testing.assert_array_equal(sd, [1, 2, 3, 4, 5])

    def test_quantiles_symmetric(self):
        data = np.random.normal(0, 1, 100)
        _, tq = probability_plot(data)
        # Should be approximately symmetric around 0
        assert pytest.approx(tq.mean(), abs=0.1) == 0.0

    def test_too_few_data_raises(self):
        with pytest.raises(ValueError):
            probability_plot([1])
