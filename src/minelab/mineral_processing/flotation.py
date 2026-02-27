"""Flotation kinetics and circuit calculations.

First-order kinetics, Kelsall two-component model, bank design,
circuit recovery, selectivity index, and kinetics fitting.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from minelab.utilities.validators import validate_positive, validate_range

# ---------------------------------------------------------------------------
# First-Order Flotation (P4-P09)
# ---------------------------------------------------------------------------


def flotation_first_order(
    r_inf: float,
    k: float,
    t: float,
) -> float:
    """First-order flotation recovery model.

    Parameters
    ----------
    r_inf : float
        Ultimate recovery (fraction, 0-1).
    k : float
        Rate constant (1/min).
    t : float
        Flotation time (min).

    Returns
    -------
    float
        Recovery (fraction, 0-1).

    Examples
    --------
    >>> round(flotation_first_order(0.95, 0.5, 10), 3)
    0.944

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.12.
    """
    validate_range(r_inf, 0, 1, "r_inf")
    validate_positive(k, "k")

    # R = R_inf * (1 - exp(-k*t))  — first-order kinetics
    return float(r_inf * (1 - np.exp(-k * t)))


# ---------------------------------------------------------------------------
# Kelsall Two-Component Model (P4-P10)
# ---------------------------------------------------------------------------


def flotation_kelsall(
    r_inf_fast: float,
    k_fast: float,
    r_inf_slow: float,
    k_slow: float,
    t: float,
) -> float:
    """Kelsall two-component flotation model.

    Parameters
    ----------
    r_inf_fast : float
        Ultimate recovery of fast-floating fraction (0-1).
    k_fast : float
        Rate constant for fast fraction (1/min).
    r_inf_slow : float
        Ultimate recovery of slow-floating fraction (0-1).
    k_slow : float
        Rate constant for slow fraction (1/min).
    t : float
        Flotation time (min).

    Returns
    -------
    float
        Total recovery (fraction, 0-1).

    Examples
    --------
    >>> round(flotation_kelsall(0.6, 2.0, 0.3, 0.2, 5), 3)
    0.8

    References
    ----------
    .. [1] Kelsall, D.F. (1961). "Application of probability in the
       assessment of flotation systems." Trans. IMM, 70, 191-204.
    """
    validate_range(r_inf_fast, 0, 1, "r_inf_fast")
    validate_range(r_inf_slow, 0, 1, "r_inf_slow")
    validate_positive(k_fast, "k_fast")
    validate_positive(k_slow, "k_slow")

    # R = R_inf_fast*(1-exp(-k_fast*t)) + R_inf_slow*(1-exp(-k_slow*t))
    r_fast = r_inf_fast * (1 - np.exp(-k_fast * t))
    r_slow = r_inf_slow * (1 - np.exp(-k_slow * t))

    return float(r_fast + r_slow)


# ---------------------------------------------------------------------------
# Flotation Bank Design (P4-P11)
# ---------------------------------------------------------------------------


def flotation_bank_design(
    recovery_target: float,
    k: float,
    cell_volume: float,
    feed_rate: float,
    r_inf: float = 1.0,
) -> dict:
    """Number of cells required for a flotation bank.

    Parameters
    ----------
    recovery_target : float
        Target recovery (fraction, 0-1).
    k : float
        Rate constant (1/min).
    cell_volume : float
        Volume of each cell (m^3).
    feed_rate : float
        Volumetric feed rate (m^3/min).
    r_inf : float
        Ultimate recovery (fraction). Default 1.0.

    Returns
    -------
    dict
        Keys: ``"n_cells"`` (int), ``"residence_time"`` (min per cell),
        ``"total_residence_time"`` (min).

    Examples
    --------
    >>> result = flotation_bank_design(0.9, 0.5, 10, 5)
    >>> result["n_cells"]
    5

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.12.
    """
    validate_range(recovery_target, 0, 1, "recovery_target")
    validate_positive(k, "k")
    validate_positive(cell_volume, "cell_volume")
    validate_positive(feed_rate, "feed_rate")
    validate_range(r_inf, 0, 1, "r_inf")

    # Residence time per cell: tau = V/Q
    tau = cell_volume / feed_rate

    # n_cells = -ln(1 - R/R_inf) / (k*tau)
    ratio = recovery_target / r_inf
    if ratio >= 1:
        ratio = 0.999  # cap to avoid log(0)
    n_cells = int(np.ceil(-np.log(1 - ratio) / (k * tau)))

    total_time = n_cells * tau

    return {
        "n_cells": n_cells,
        "residence_time": float(tau),
        "total_residence_time": float(total_time),
    }


# ---------------------------------------------------------------------------
# Flotation Circuit (P4-P12)
# ---------------------------------------------------------------------------


def flotation_circuit(
    rougher_r: float,
    cleaner_r: float,
    scavenger_r: float = 0.0,
) -> dict:
    """Overall recovery for a rougher-scavenger-cleaner circuit.

    Parameters
    ----------
    rougher_r : float
        Rougher recovery (fraction, 0-1).
    cleaner_r : float
        Cleaner recovery (fraction, 0-1).
    scavenger_r : float
        Scavenger recovery on rougher tailings (fraction, 0-1).
        Default 0.0 (no scavenger).

    Returns
    -------
    dict
        Keys: ``"overall_recovery"`` (fraction),
        ``"rougher_cleaner_recovery"`` (fraction).

    Examples
    --------
    >>> result = flotation_circuit(0.9, 0.8, 0.5)
    >>> result["overall_recovery"] > 0
    True

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.12.
    """
    validate_range(rougher_r, 0, 1, "rougher_r")
    validate_range(cleaner_r, 0, 1, "cleaner_r")
    validate_range(scavenger_r, 0, 1, "scavenger_r")

    # Rougher-cleaner overall
    rc_recovery = rougher_r * cleaner_r

    # With scavenger: additional recovery from rougher tails
    scav_recovery = (1 - rougher_r) * scavenger_r * cleaner_r
    overall = rc_recovery + scav_recovery

    return {
        "overall_recovery": float(overall),
        "rougher_cleaner_recovery": float(rc_recovery),
    }


# ---------------------------------------------------------------------------
# Selectivity Index (P4-P13)
# ---------------------------------------------------------------------------


def selectivity_index(
    recovery_mineral: float,
    recovery_gangue: float,
) -> float:
    """Gaudin selectivity index.

    Parameters
    ----------
    recovery_mineral : float
        Recovery of valuable mineral (fraction, 0-1).
    recovery_gangue : float
        Recovery of gangue (fraction, 0-1).

    Returns
    -------
    float
        Selectivity index SI.

    Examples
    --------
    >>> round(selectivity_index(0.9, 0.1), 1)
    81.0

    References
    ----------
    .. [1] Gaudin, A.M. (1939). Principles of Mineral Dressing. McGraw-Hill.
    """
    validate_range(recovery_mineral, 0.001, 0.999, "recovery_mineral")
    validate_range(recovery_gangue, 0.001, 0.999, "recovery_gangue")

    # SI = Rm*(1-Rg) / (Rg*(1-Rm))  — Gaudin 1939
    si = (recovery_mineral * (1 - recovery_gangue)) / (recovery_gangue * (1 - recovery_mineral))

    return float(si)


# ---------------------------------------------------------------------------
# Flotation Kinetics Fit (P4-P14)
# ---------------------------------------------------------------------------


def flotation_kinetics_fit(
    times: np.ndarray,
    recoveries: np.ndarray,
) -> dict:
    """Fit first-order flotation kinetics to experimental data.

    Parameters
    ----------
    times : np.ndarray
        Flotation times (min).
    recoveries : np.ndarray
        Corresponding recoveries (fractions, 0-1).

    Returns
    -------
    dict
        Keys: ``"r_inf"`` (ultimate recovery), ``"k"`` (rate constant),
        ``"r_squared"`` (coefficient of determination).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0, 1, 2, 5, 10, 20])
    >>> r = 0.95 * (1 - np.exp(-0.5 * t))
    >>> result = flotation_kinetics_fit(t, r)
    >>> round(result["r_inf"], 2)
    0.95

    References
    ----------
    .. [1] Standard practice; nonlinear regression on R = R_inf*(1-exp(-k*t)).
    """
    times = np.asarray(times, dtype=float)
    recoveries = np.asarray(recoveries, dtype=float)

    def _model(t, r_inf, k):
        return r_inf * (1 - np.exp(-k * t))

    # Initial guesses
    p0 = [float(np.max(recoveries)), 0.5]

    popt, _ = curve_fit(
        _model,
        times,
        recoveries,
        p0=p0,
        bounds=([0, 0], [1, 100]),
        maxfev=5000,
    )

    r_inf_fit, k_fit = popt

    # R-squared calculation
    y_pred = _model(times, r_inf_fit, k_fit)
    ss_res = np.sum((recoveries - y_pred) ** 2)
    ss_tot = np.sum((recoveries - np.mean(recoveries)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "r_inf": float(r_inf_fit),
        "k": float(k_fit),
        "r_squared": float(r_squared),
    }
