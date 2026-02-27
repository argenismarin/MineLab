"""Variogram model fitting tools.

Provides weighted least-squares fitting, manual model creation, and
automatic model selection for experimental variograms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from minelab.geostatistics.variogram_models import (
    exponential,
    gaussian,
    spherical,
)
from minelab.utilities.validators import validate_array


@dataclass
class VariogramModel:
    """Container for a fitted or manually defined variogram model.

    Attributes
    ----------
    model_type : str
        Model name (``"spherical"``, ``"exponential"``, ``"gaussian"``).
    nugget : float
        Nugget variance.
    sill : float
        Total sill (nugget + partial sill).
    range_a : float
        Practical range.
    rmse : float
        Root mean square error of fit (0.0 for manual models).
    """

    model_type: str
    nugget: float
    sill: float
    range_a: float
    rmse: float = 0.0
    _model_funcs: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._model_funcs = {
            "spherical": spherical,
            "exponential": exponential,
            "gaussian": gaussian,
        }

    def predict(self, h: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the variogram model at lag distance(s) h."""
        func = self._model_funcs[self.model_type]
        return func(h, self.nugget, self.sill, self.range_a)


def fit_variogram_wls(
    exp_lags: np.ndarray,
    exp_sv: np.ndarray,
    model_type: str = "spherical",
    n_pairs: np.ndarray | None = None,
) -> VariogramModel:
    """Fit a variogram model by weighted least squares.

    Parameters
    ----------
    exp_lags : np.ndarray
        Experimental lag distances.
    exp_sv : np.ndarray
        Experimental semivariance values.
    model_type : str
        One of ``"spherical"``, ``"exponential"``, ``"gaussian"``.
    n_pairs : np.ndarray or None
        Number of pairs per lag bin. Used for Cressie weights
        ``w_i = N_i / γ(h_i)²``. If None, equal weights.

    Returns
    -------
    VariogramModel
        Fitted model with nugget, sill, range_a, and RMSE.

    Examples
    --------
    >>> lags = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    >>> sv = np.array([2.1, 4.0, 5.5, 6.8, 7.5, 8.0, 8.5, 8.8, 9.0, 9.2])
    >>> model = fit_variogram_wls(lags, sv, "spherical")
    >>> model.model_type
    'spherical'

    References
    ----------
    .. [1] Cressie, N. (1985). "Fitting variogram models by weighted least
       squares." J. Int. Assoc. Math. Geol., 17(5), 563-586.
    """
    model_funcs = {
        "spherical": spherical,
        "exponential": exponential,
        "gaussian": gaussian,
    }
    if model_type not in model_funcs:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {list(model_funcs.keys())}"
        )

    exp_lags = validate_array(exp_lags, "exp_lags")
    exp_sv = validate_array(exp_sv, "exp_sv")

    # Remove NaN entries
    valid = ~np.isnan(exp_sv) & ~np.isnan(exp_lags)
    lags = exp_lags[valid]
    sv = exp_sv[valid]

    if n_pairs is not None:
        n_pairs = validate_array(n_pairs, "n_pairs")[valid]

    if len(lags) < 3:
        raise ValueError("Need at least 3 valid lag bins for fitting.")

    func = model_funcs[model_type]

    # Initial guesses
    nugget0 = max(0.0, float(sv[0]) * 0.5)
    sill0 = float(np.max(sv))
    range0 = float(lags[np.argmax(sv >= 0.95 * sill0)] if np.any(sv >= 0.95 * sill0) else lags[-1])

    def residuals(params: np.ndarray) -> np.ndarray:
        nug, sil, rng = params
        nug = max(nug, 0.0)
        sil = max(sil, nug + 1e-10)
        rng = max(rng, 1e-10)
        predicted = np.asarray(func(lags, nug, sil, rng), dtype=float)

        if n_pairs is not None:
            # Cressie (1985) weights: Ni / γ_model(hi)^2
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(predicted > 0, n_pairs / predicted**2, 1.0)
            weights = np.sqrt(weights)
        else:
            weights = np.ones_like(lags)

        return weights * (sv - predicted)

    result = least_squares(
        residuals,
        x0=[nugget0, sill0, range0],
        bounds=([0, 0, 1e-10], [np.inf, np.inf, np.inf]),
        method="trf",
    )

    nugget_fit, sill_fit, range_fit = result.x
    predicted = np.asarray(func(lags, nugget_fit, sill_fit, range_fit), dtype=float)
    rmse = float(np.sqrt(np.mean((sv - predicted) ** 2)))

    return VariogramModel(
        model_type=model_type,
        nugget=float(nugget_fit),
        sill=float(sill_fit),
        range_a=float(range_fit),
        rmse=rmse,
    )


def fit_variogram_manual(
    model_type: str,
    nugget: float,
    sill: float,
    range_a: float,
) -> VariogramModel:
    """Create a VariogramModel from manual parameters.

    Parameters
    ----------
    model_type : str
        One of ``"spherical"``, ``"exponential"``, ``"gaussian"``.
    nugget : float
        Nugget variance (>= 0).
    sill : float
        Total sill (> 0).
    range_a : float
        Practical range (> 0).

    Returns
    -------
    VariogramModel
        Model object with rmse = 0.0.

    Examples
    --------
    >>> m = fit_variogram_manual("spherical", 0, 10, 100)
    >>> m.predict(50)
    6.875

    References
    ----------
    .. [1] Standard geostatistical practice.
    """
    valid_types = {"spherical", "exponential", "gaussian"}
    if model_type not in valid_types:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {sorted(valid_types)}")

    return VariogramModel(
        model_type=model_type,
        nugget=float(nugget),
        sill=float(sill),
        range_a=float(range_a),
        rmse=0.0,
    )


def auto_fit(
    exp_lags: np.ndarray,
    exp_sv: np.ndarray,
    n_pairs: np.ndarray | None = None,
    models: list[str] | None = None,
) -> VariogramModel:
    """Automatically fit the best variogram model by RMSE.

    Parameters
    ----------
    exp_lags : np.ndarray
        Experimental lag distances.
    exp_sv : np.ndarray
        Experimental semivariance values.
    n_pairs : np.ndarray or None
        Number of pairs per lag bin.
    models : list of str or None
        Models to try. Default: ``["spherical", "exponential", "gaussian"]``.

    Returns
    -------
    VariogramModel
        Best-fit model (lowest RMSE).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_models import spherical
    >>> lags = np.arange(10, 110, 10, dtype=float)
    >>> sv = np.array([spherical(h, 0, 10, 80) for h in lags]) + 0.1
    >>> best = auto_fit(lags, sv)
    >>> best.model_type
    'spherical'

    References
    ----------
    .. [1] Standard geostatistical practice.
    """
    if models is None:
        models = ["spherical", "exponential", "gaussian"]

    best: VariogramModel | None = None
    for mt in models:
        try:
            fitted = fit_variogram_wls(exp_lags, exp_sv, mt, n_pairs)
            if best is None or fitted.rmse < best.rmse:
                best = fitted
        except Exception:
            continue

    if best is None:
        raise RuntimeError("All model fits failed.")

    return best
