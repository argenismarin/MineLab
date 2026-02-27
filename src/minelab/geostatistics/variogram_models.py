"""Theoretical variogram models for geostatistical analysis.

Provides standard variogram model functions: spherical, exponential,
gaussian, power, nugget effect, hole effect, and nested combinations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minelab.utilities.validators import validate_non_negative, validate_positive


def spherical(
    h: float | np.ndarray,
    nugget: float,
    sill: float,
    range_a: float,
) -> float | np.ndarray:
    """Spherical variogram model.

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance (C0).
    sill : float
        Total sill (C0 + C). The partial sill C = sill - nugget.
    range_a : float
        Range parameter (a) beyond which the variogram reaches the sill.

    Returns
    -------
    float or np.ndarray
        Semivariance value(s).

    Examples
    --------
    >>> spherical(50, 0, 10, 100)
    6.875
    >>> spherical(100, 0, 10, 100)
    10.0

    References
    ----------
    .. [1] Isaaks, E.H. & Srivastava, R.M. (1989). "An Introduction to
       Applied Geostatistics." Oxford University Press, Ch. 12.
    """
    validate_non_negative(nugget, "nugget")
    validate_positive(sill, "sill")
    validate_positive(range_a, "range_a")

    h = np.asarray(h, dtype=float)
    partial_sill = sill - nugget  # C

    # γ(h) = C0 + C * [1.5*(h/a) - 0.5*(h/a)^3]  for h <= a
    # γ(h) = C0 + C                                 for h > a
    # γ(0) = 0
    hr = np.clip(h / range_a, 0.0, 1.0)
    gamma = np.where(
        h == 0,
        0.0,
        nugget + partial_sill * (1.5 * hr - 0.5 * hr**3),
    )
    result = np.where(h >= range_a, sill, gamma)
    result = np.where(h == 0, 0.0, result)

    return float(result) if result.ndim == 0 else result


def exponential(
    h: float | np.ndarray,
    nugget: float,
    sill: float,
    range_a: float,
) -> float | np.ndarray:
    """Exponential variogram model.

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance (C0).
    sill : float
        Total sill (C0 + C).
    range_a : float
        Practical range (distance at ~95% of sill).

    Returns
    -------
    float or np.ndarray
        Semivariance value(s).

    Examples
    --------
    >>> round(exponential(100, 0, 10, 100), 3)
    9.502

    References
    ----------
    .. [1] Cressie, N. (1993). "Statistics for Spatial Data." Wiley.
    """
    validate_non_negative(nugget, "nugget")
    validate_positive(sill, "sill")
    validate_positive(range_a, "range_a")

    h = np.asarray(h, dtype=float)
    partial_sill = sill - nugget

    # γ(h) = C0 + C * [1 - exp(-3h/a)]
    gamma = np.where(
        h == 0,
        0.0,
        nugget + partial_sill * (1.0 - np.exp(-3.0 * h / range_a)),
    )

    return float(gamma) if gamma.ndim == 0 else gamma


def gaussian(
    h: float | np.ndarray,
    nugget: float,
    sill: float,
    range_a: float,
) -> float | np.ndarray:
    """Gaussian variogram model.

    Parabolic behavior at the origin (very smooth spatial continuity).

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance (C0).
    sill : float
        Total sill (C0 + C).
    range_a : float
        Practical range (distance at ~95% of sill).

    Returns
    -------
    float or np.ndarray
        Semivariance value(s).

    Examples
    --------
    >>> round(gaussian(100, 0, 10, 100), 3)
    9.502

    References
    ----------
    .. [1] Cressie, N. (1993). "Statistics for Spatial Data." Wiley.
    """
    validate_non_negative(nugget, "nugget")
    validate_positive(sill, "sill")
    validate_positive(range_a, "range_a")

    h = np.asarray(h, dtype=float)
    partial_sill = sill - nugget

    # γ(h) = C0 + C * [1 - exp(-3*h²/a²)]
    gamma = np.where(
        h == 0,
        0.0,
        nugget + partial_sill * (1.0 - np.exp(-3.0 * (h / range_a) ** 2)),
    )

    return float(gamma) if gamma.ndim == 0 else gamma


def power(
    h: float | np.ndarray,
    nugget: float,
    slope: float,
    exponent: float,
) -> float | np.ndarray:
    """Power variogram model (unbounded).

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance (C0).
    slope : float
        Slope coefficient (b > 0).
    exponent : float
        Power exponent (0 < omega < 2).

    Returns
    -------
    float or np.ndarray
        Semivariance value(s).

    Examples
    --------
    >>> power(10, 0, 1.5, 1.0)
    15.0

    References
    ----------
    .. [1] Chilès, J.-P. & Delfiner, P. (2012). "Geostatistics: Modeling
       Spatial Uncertainty." 2nd ed., Wiley.
    """
    validate_non_negative(nugget, "nugget")
    validate_positive(slope, "slope")
    if not 0 < exponent < 2:
        raise ValueError(f"'exponent' must be in (0, 2), got {exponent}.")

    h = np.asarray(h, dtype=float)

    # γ(h) = C0 + b * h^ω
    gamma = np.where(h == 0, 0.0, nugget + slope * np.abs(h) ** exponent)

    return float(gamma) if gamma.ndim == 0 else gamma


def nugget_effect(
    h: float | np.ndarray,
    nugget: float,
) -> float | np.ndarray:
    """Pure nugget effect model.

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance.

    Returns
    -------
    float or np.ndarray
        Semivariance value(s): 0 at h=0, nugget for h > 0.

    Examples
    --------
    >>> nugget_effect(0, 5.0)
    0.0
    >>> nugget_effect(10, 5.0)
    5.0

    References
    ----------
    .. [1] Journel, A.G. & Huijbregts, C.J. (1978). "Mining Geostatistics."
       Academic Press.
    """
    validate_positive(nugget, "nugget")

    h = np.asarray(h, dtype=float)

    # γ(0) = 0, γ(h>0) = nugget
    gamma = np.where(h == 0, 0.0, nugget)

    return float(gamma) if gamma.ndim == 0 else gamma


def hole_effect(
    h: float | np.ndarray,
    nugget: float,
    sill: float,
    range_a: float,
) -> float | np.ndarray:
    """Hole-effect (periodic) variogram model.

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    nugget : float
        Nugget variance (C0).
    sill : float
        Total sill (C0 + C).
    range_a : float
        Range parameter controlling the period of oscillation.

    Returns
    -------
    float or np.ndarray
        Semivariance value(s).

    Examples
    --------
    >>> round(hole_effect(50, 0, 10, 100), 3)
    3.633

    References
    ----------
    .. [1] Chilès, J.-P. & Delfiner, P. (2012). "Geostatistics: Modeling
       Spatial Uncertainty." 2nd ed., Wiley.
    """
    validate_non_negative(nugget, "nugget")
    validate_positive(sill, "sill")
    validate_positive(range_a, "range_a")

    h = np.asarray(h, dtype=float)
    partial_sill = sill - nugget

    # γ(h) = C0 + C * [1 - sin(π*h/a) / (π*h/a)]
    # At h=0, sin(x)/x → 1, so γ(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.pi * h / range_a
        sinc_term = np.where(h == 0, 1.0, np.sin(ratio) / ratio)

    gamma = np.where(h == 0, 0.0, nugget + partial_sill * (1.0 - sinc_term))

    return float(gamma) if gamma.ndim == 0 else gamma


def nested_model(
    h: float | np.ndarray,
    structures: list[dict],
) -> float | np.ndarray:
    """Nested (composite) variogram model as sum of basic structures.

    Parameters
    ----------
    h : float or np.ndarray
        Lag distance(s).
    structures : list of dict
        Each dict must have ``"model"`` (callable or str) and the
        keyword arguments for that model **excluding** ``h``.
        Example::

            [{"model": "spherical", "nugget": 0, "sill": 5, "range_a": 50},
             {"model": "exponential", "nugget": 0, "sill": 5, "range_a": 100}]

    Returns
    -------
    float or np.ndarray
        Sum of all structure semivariances.

    Examples
    --------
    >>> structs = [
    ...     {"model": "spherical", "nugget": 0, "sill": 5, "range_a": 100},
    ...     {"model": "exponential", "nugget": 0, "sill": 5, "range_a": 100},
    ... ]
    >>> round(nested_model(50, structs), 3)
    8.189

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press.
    """
    model_map: dict[str, Callable] = {
        "spherical": spherical,
        "exponential": exponential,
        "gaussian": gaussian,
        "power": power,
        "nugget": nugget_effect,
        "hole_effect": hole_effect,
    }

    if not structures:
        raise ValueError("'structures' must contain at least one model.")

    h = np.asarray(h, dtype=float)
    total = np.zeros_like(h)

    for i, s in enumerate(structures):
        s = dict(s)  # copy to avoid mutating input
        model = s.pop("model")
        if isinstance(model, str):
            if model not in model_map:
                raise ValueError(
                    f"Unknown model '{model}' in structure {i}. "
                    f"Choose from: {list(model_map.keys())}"
                )
            model = model_map[model]
        total = total + np.asarray(model(h, **s))

    return float(total) if total.ndim == 0 else total
