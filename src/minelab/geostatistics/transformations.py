"""Geostatistical data transformations.

Normal score, back-transform, Gaussian anamorphosis, indicator transform,
and lognormal transform functions.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats
from scipy.interpolate import interp1d
from scipy.special import hermite

from minelab.utilities.validators import validate_array, validate_positive


def normal_score_transform(
    data: np.ndarray,
) -> dict:
    """Rank-based normal score transformation to N(0,1).

    Parameters
    ----------
    data : np.ndarray
        Original data values, shape (n,).

    Returns
    -------
    dict
        Keys: ``"transformed"`` (normal scores), ``"transform_table"``
        (2-column array mapping original → transformed for back-transform).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.lognormal(0, 1, 1000)
    >>> result = normal_score_transform(data)
    >>> abs(np.mean(result["transformed"])) < 0.1
    True

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 7.
    """
    data = validate_array(data, "data")
    n = len(data)

    # Rank-based transform
    order = np.argsort(data)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)

    # Van der Waerden scores: Φ⁻¹(rank / (n+1))
    probs = ranks / (n + 1)
    ns_values = sp_stats.norm.ppf(probs)

    # Build transform table sorted by original values
    table = np.column_stack([data[order], ns_values[order]])

    return {"transformed": ns_values, "transform_table": table}


def back_transform(
    ns_values: np.ndarray,
    transform_table: np.ndarray,
) -> np.ndarray:
    """Back-transform normal scores to original data space.

    Parameters
    ----------
    ns_values : np.ndarray
        Normal score values to back-transform.
    transform_table : np.ndarray
        Two-column array from ``normal_score_transform``: col 0 = original,
        col 1 = normal scores (sorted by original).

    Returns
    -------
    np.ndarray
        Back-transformed values in original data space.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.lognormal(0, 1, 100)
    >>> result = normal_score_transform(data)
    >>> bt = back_transform(result["transformed"], result["transform_table"])
    >>> np.allclose(np.sort(bt), np.sort(data), rtol=1e-6)
    True

    References
    ----------
    .. [1] Deutsch, C.V. & Journel, A.G. (1998). "GSLIB." Oxford Univ. Press.
    """
    ns_values = np.asarray(ns_values, dtype=float)
    table = np.asarray(transform_table, dtype=float)

    original_sorted = table[:, 0]
    ns_sorted = table[:, 1]

    # Linear interpolation from NS space back to original
    interpolator = interp1d(
        ns_sorted,
        original_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=(original_sorted[0], original_sorted[-1]),
    )

    return interpolator(ns_values)


def gaussian_anamorphosis(
    data: np.ndarray,
    n_hermite: int = 30,
) -> dict:
    """Gaussian anamorphosis using Hermite polynomial expansion.

    Parameters
    ----------
    data : np.ndarray
        Original data values.
    n_hermite : int
        Number of Hermite polynomials (default 30).

    Returns
    -------
    dict
        Keys: ``"coefficients"`` (Hermite polynomial coefficients),
        ``"transform_table"`` (for back-transform),
        ``"transformed"`` (normal scores).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.lognormal(0, 1, 500)
    >>> result = gaussian_anamorphosis(data, n_hermite=20)
    >>> len(result["coefficients"]) == 20
    True

    References
    ----------
    .. [1] Chilès, J.-P. & Delfiner, P. (2012). "Geostatistics: Modeling
       Spatial Uncertainty." 2nd ed., Wiley, Ch. 6.
    """
    validate_positive(n_hermite, "n_hermite")
    data = validate_array(data, "data")

    # First, get normal scores
    ns_result = normal_score_transform(data)
    y = ns_result["transformed"]

    # Sort by normal score for coefficient computation
    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    z_sorted = data[sort_idx]

    # Compute Hermite polynomial coefficients
    # φ_k(y) = He_k(y), probabilist's Hermite polynomials
    # c_k = (1/n) Σ z_i * He_k(y_i) / k!
    coeffs = np.zeros(n_hermite)
    for k in range(n_hermite):
        he_k = hermite(k)  # physicist's Hermite polynomial
        # Probabilist's He_k(y) = 2^(-k/2) * H_k(y/√2) where H_k is physicist's
        he_vals = he_k(y_sorted / np.sqrt(2)) / (np.sqrt(2) ** k)
        coeffs[k] = np.mean(z_sorted * he_vals) / math.factorial(k)

    return {
        "coefficients": coeffs,
        "transform_table": ns_result["transform_table"],
        "transformed": y,
    }


def indicator_transform(
    data: np.ndarray,
    cutoffs: np.ndarray | list[float],
) -> np.ndarray:
    """Indicator transformation at multiple cutoff thresholds.

    Parameters
    ----------
    data : np.ndarray
        Data values, shape (n,).
    cutoffs : array-like
        Cutoff thresholds. For each cutoff, indicator = 1 if data <= cutoff.

    Returns
    -------
    np.ndarray
        Indicator matrix, shape (n, len(cutoffs)). Each column is the
        binary indicator for the corresponding cutoff.

    Examples
    --------
    >>> data = np.array([1, 3, 5, 7, 9])
    >>> ind = indicator_transform(data, [4, 6])
    >>> ind[:, 0].tolist()
    [1, 1, 0, 0, 0]

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press.
    """
    data = validate_array(data, "data")
    cutoffs = np.asarray(cutoffs, dtype=float)
    if cutoffs.ndim == 0:
        cutoffs = cutoffs.reshape(1)

    # indicator[i, k] = 1 if data[i] <= cutoffs[k]
    indicators = (data[:, np.newaxis] <= cutoffs[np.newaxis, :]).astype(int)

    return indicators


def lognormal_transform(
    data: np.ndarray,
) -> dict:
    """Log-transform with summary statistics.

    Parameters
    ----------
    data : np.ndarray
        Strictly positive data values.

    Returns
    -------
    dict
        Keys: ``"log_values"`` (natural log), ``"mean_log"`` (mean of logs),
        ``"var_log"`` (variance of logs), ``"is_lognormal"`` (bool,
        Shapiro-Wilk p > 0.05 on logs).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.lognormal(2, 0.5, 200)
    >>> result = lognormal_transform(data)
    >>> result["is_lognormal"]
    True

    References
    ----------
    .. [1] Isaaks, E.H. & Srivastava, R.M. (1989). "An Introduction to
       Applied Geostatistics." Oxford University Press.
    """
    data = validate_array(data, "data")
    if np.any(data <= 0):
        raise ValueError("All data values must be strictly positive for log transform.")

    log_vals = np.log(data)
    mean_log = float(np.mean(log_vals))
    var_log = float(np.var(log_vals, ddof=1))

    # Test normality of log-transformed data
    n = len(log_vals)
    if n >= 8:
        _, p_value = sp_stats.shapiro(log_vals[:5000] if n > 5000 else log_vals)
        is_lognormal = bool(p_value > 0.05)
    else:
        is_lognormal = False

    return {
        "log_values": log_vals,
        "mean_log": mean_log,
        "var_log": var_log,
        "is_lognormal": is_lognormal,
    }
