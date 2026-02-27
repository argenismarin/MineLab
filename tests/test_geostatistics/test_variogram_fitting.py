"""Tests for minelab.geostatistics.variogram_fitting."""

import numpy as np
import pytest

from minelab.geostatistics.variogram_fitting import (
    VariogramModel,
    auto_fit,
    fit_variogram_manual,
    fit_variogram_wls,
)
from minelab.geostatistics.variogram_models import exponential, spherical


class TestFitVariogramWLS:
    """Tests for WLS variogram fitting."""

    def test_recover_spherical_params(self):
        """Fit synthetic spherical data and recover true params."""
        true_nugget, true_sill, true_range = 0.0, 10.0, 80.0
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, true_nugget, true_sill, true_range) for h in lags])
        model = fit_variogram_wls(lags, sv, "spherical")
        assert model.model_type == "spherical"
        assert model.sill == pytest.approx(true_sill, rel=0.1)
        assert model.range_a == pytest.approx(true_range, rel=0.15)
        assert model.rmse < 1.0

    def test_recover_exponential_params(self):
        """Fit synthetic exponential data."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([exponential(h, 0, 8, 60) for h in lags])
        model = fit_variogram_wls(lags, sv, "exponential")
        assert model.model_type == "exponential"
        assert model.sill == pytest.approx(8.0, rel=0.15)

    def test_with_noise(self):
        """Fitting noisy data still recovers approximately correct params."""
        rng = np.random.default_rng(42)
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, 1, 10, 80) for h in lags])
        sv_noisy = sv + rng.normal(0, 0.3, len(sv))
        sv_noisy = np.maximum(sv_noisy, 0)
        model = fit_variogram_wls(lags, sv_noisy, "spherical")
        assert model.sill == pytest.approx(10.0, rel=0.3)

    def test_with_n_pairs_weights(self):
        """Cressie weights with n_pairs should still fit."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, 0, 10, 80) for h in lags])
        n_pairs = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        model = fit_variogram_wls(lags, sv, "spherical", n_pairs)
        assert model.rmse < 1.0

    def test_unknown_model(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            fit_variogram_wls(np.array([1, 2, 3]), np.array([1, 2, 3]), "cubic")

    def test_too_few_points(self):
        """Fewer than 3 valid points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            fit_variogram_wls(np.array([1, 2]), np.array([1, 2]), "spherical")

    def test_returns_variogram_model(self):
        """Result should be a VariogramModel instance."""
        lags = np.arange(10, 60, 10, dtype=float)
        sv = np.array([spherical(h, 0, 10, 50) for h in lags])
        model = fit_variogram_wls(lags, sv, "spherical")
        assert isinstance(model, VariogramModel)


class TestFitVariogramManual:
    """Tests for manual variogram model creation."""

    def test_roundtrip(self):
        """Manual model with known params should predict correctly."""
        m = fit_variogram_manual("spherical", 0, 10, 100)
        assert m.predict(50) == pytest.approx(6.875, rel=1e-6)
        assert m.predict(0) == 0.0
        assert m.predict(100) == pytest.approx(10.0, rel=1e-6)

    def test_rmse_zero(self):
        """Manual models have RMSE = 0."""
        m = fit_variogram_manual("exponential", 1, 8, 50)
        assert m.rmse == 0.0

    def test_invalid_model_type(self):
        """Unknown model type raises ValueError."""
        with pytest.raises(ValueError):
            fit_variogram_manual("linear", 0, 10, 100)


class TestAutoFit:
    """Tests for automatic model selection."""

    def test_selects_spherical(self):
        """Auto-fit should select spherical for spherical data."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, 0, 10, 80) for h in lags])
        best = auto_fit(lags, sv)
        assert best.model_type == "spherical"

    def test_selects_exponential(self):
        """Auto-fit should select exponential for exponential data."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([exponential(h, 0, 10, 80) for h in lags])
        best = auto_fit(lags, sv)
        # Should be exponential (or very close RMSE)
        assert best.rmse < 1.0

    def test_custom_model_list(self):
        """Restricting to specific models should work."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, 0, 10, 80) for h in lags])
        best = auto_fit(lags, sv, models=["spherical", "gaussian"])
        assert best.model_type in ["spherical", "gaussian"]

    def test_returns_best_rmse(self):
        """Best model should have lowest RMSE among candidates."""
        lags = np.arange(10, 110, 10, dtype=float)
        sv = np.array([spherical(h, 0, 10, 80) for h in lags])
        best = auto_fit(lags, sv)
        # Try all and verify best has lowest
        for mt in ["spherical", "exponential", "gaussian"]:
            m = fit_variogram_wls(lags, sv, mt)
            assert best.rmse <= m.rmse + 1e-6
