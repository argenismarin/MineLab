"""Fan selection and performance analysis for mine ventilation.

This module provides tools for determining fan operating points,
computing fan power, and combining fan characteristic curves for
series and parallel configurations.

References
----------
.. [1] McPherson, M.J. (1993). *Subsurface Ventilation and Environmental
       Engineering*, 1st ed. Chapman & Hall, Chapter 10.
"""

from __future__ import annotations

import numpy as np  # noqa: I001

from minelab.utilities.validators import validate_non_negative, validate_positive

# ---------------------------------------------------------------------------
# Fan operating point
# ---------------------------------------------------------------------------


def fan_operating_point(
    fan_curve_q: np.ndarray,
    fan_curve_p: np.ndarray,
    system_resistance: float,
) -> dict:
    """Find the operating point at the intersection of fan and system curves.

    The fan curve is given as discrete (Q, P) pairs.  The system curve is
    parabolic:

    .. math::

        P_{\\text{sys}} = R \\cdot Q^2

    The intersection is found by linear interpolation between the two
    nearest bracketing points.

    Parameters
    ----------
    fan_curve_q : numpy.ndarray
        Volume flow rate values on the fan characteristic curve (m^3/s).
        Must be monotonically increasing with at least 2 elements.
    fan_curve_p : numpy.ndarray
        Corresponding fan pressure values (Pa).  Same length as
        *fan_curve_q*.
    system_resistance : float
        System resistance (Ns^2/m^8).  Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"Q_operating"`` : float -- Operating airflow (m^3/s).
        - ``"P_operating"`` : float -- Operating pressure (Pa).

    Raises
    ------
    ValueError
        If input arrays are too short, mismatched, or no intersection is
        found within the data range.

    Examples
    --------
    >>> Q = np.array([0, 20, 40, 60, 80, 100])
    >>> P = np.array([3000, 2800, 2400, 1800, 1000, 0])
    >>> result = fan_operating_point(Q, P, system_resistance=0.3)
    >>> 60 < result["Q_operating"] < 80
    True

    References
    ----------
    .. [1] McPherson (1993), Ch. 10, Sec. 10.3.
    """
    fq = np.asarray(fan_curve_q, dtype=float).ravel()
    fp = np.asarray(fan_curve_p, dtype=float).ravel()

    if fq.size < 2:
        raise ValueError("'fan_curve_q' must have at least 2 elements.")
    if fq.size != fp.size:
        raise ValueError(
            "'fan_curve_q' and 'fan_curve_p' must have the same length. "
            f"Got {fq.size} and {fp.size}."
        )
    validate_positive(system_resistance, "system_resistance")

    # Compute the difference: fan_P - system_P at each Q
    sys_p = system_resistance * fq**2
    diff = fp - sys_p

    # Look for a sign change in diff (fan curve crosses system curve)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            # Linear interpolation between points i and i+1
            if diff[i] == diff[i + 1]:
                # Exactly equal at both points (degenerate)
                q_op = fq[i]
            else:
                frac = diff[i] / (diff[i] - diff[i + 1])
                q_op = fq[i] + frac * (fq[i + 1] - fq[i])
            p_op = system_resistance * q_op**2
            return {
                "Q_operating": float(q_op),
                "P_operating": float(p_op),
            }

    raise ValueError(
        "No intersection found between fan curve and system curve "
        "within the provided data range. Check that the fan curve "
        "spans the expected operating range."
    )


# ---------------------------------------------------------------------------
# Fan power
# ---------------------------------------------------------------------------


def fan_power(airflow: float, pressure: float, efficiency: float) -> float:
    """Compute the input power required to drive a fan.

    .. math::

        \\text{Power} = \\frac{Q \\cdot P}{\\eta}

    Parameters
    ----------
    airflow : float
        Volume airflow rate Q (m^3/s).  Must be non-negative.
    pressure : float
        Fan total pressure P (Pa).  Must be non-negative.
    efficiency : float
        Fan total efficiency (dimensionless, 0 < eta <= 1).

    Returns
    -------
    float
        Fan input power in watts (W).

    Raises
    ------
    ValueError
        If *airflow* or *pressure* is negative, or *efficiency* is not in
        (0, 1].

    Examples
    --------
    >>> round(fan_power(50, 2000, 0.7), 1)
    142857.1

    References
    ----------
    .. [1] McPherson (1993), Ch. 10, Eq. 10.1.
    """
    validate_non_negative(airflow, "airflow")
    validate_non_negative(pressure, "pressure")
    validate_positive(efficiency, "efficiency")
    if efficiency > 1.0:
        raise ValueError(f"'efficiency' must be <= 1.0, got {efficiency}.")
    return airflow * pressure / efficiency


# ---------------------------------------------------------------------------
# Fans in series / parallel
# ---------------------------------------------------------------------------


def fans_in_series_parallel(
    fan_curves: list[dict],
    configuration: str,
) -> dict:
    """Compute combined fan characteristic curves for multi-fan installations.

    For **series** configuration, pressures are summed at each common
    airflow rate.  For **parallel** configuration, airflows are summed at
    each common pressure.

    Parameters
    ----------
    fan_curves : list of dict
        Each dictionary contains:

        - ``"Q"`` : numpy.ndarray -- Airflow values (m^3/s).
        - ``"P"`` : numpy.ndarray -- Pressure values (Pa).

    configuration : str
        Either ``"series"`` or ``"parallel"``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"Q"`` : numpy.ndarray -- Combined airflow values (m^3/s).
        - ``"P"`` : numpy.ndarray -- Combined pressure values (Pa).

    Raises
    ------
    ValueError
        If *fan_curves* is empty or *configuration* is unrecognised.

    Notes
    -----
    A common set of evaluation points is generated from the overlapping
    range of all input curves.  Individual fan curves are linearly
    interpolated onto this common grid before combining.

    Examples
    --------
    Series combination of two identical fans doubles the pressure:

    >>> fan = {"Q": np.array([0, 50, 100]), "P": np.array([2000, 1000, 0])}
    >>> result = fans_in_series_parallel([fan, fan], "series")
    >>> float(result["P"][0])
    4000.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 10, Sec. 10.5.
    """
    if not fan_curves:
        raise ValueError("'fan_curves' must contain at least one fan curve.")

    config = configuration.lower()
    if config not in ("series", "parallel"):
        raise ValueError(f"'configuration' must be 'series' or 'parallel', got '{configuration}'.")

    # Validate and convert all curves
    curves = []
    for i, fc in enumerate(fan_curves):
        if "Q" not in fc or "P" not in fc:
            raise ValueError(f"Fan curve {i} must contain 'Q' and 'P' keys.")
        q = np.asarray(fc["Q"], dtype=float).ravel()
        p = np.asarray(fc["P"], dtype=float).ravel()
        if q.size < 2 or q.size != p.size:
            raise ValueError(
                f"Fan curve {i}: 'Q' and 'P' must have the same length "
                f"(>= 2). Got Q={q.size}, P={p.size}."
            )
        curves.append((q, p))

    n_points = 100

    if config == "series":
        # Sum pressures at the same Q
        # Common Q range: intersection of all fans' Q ranges
        q_min = max(c[0].min() for c in curves)
        q_max = min(c[0].max() for c in curves)
        if q_min >= q_max:
            raise ValueError(
                "Fan curves have no overlapping airflow range for series combination."
            )
        q_common = np.linspace(q_min, q_max, n_points)
        p_total = np.zeros(n_points, dtype=float)
        for q_fan, p_fan in curves:
            p_interp = np.interp(q_common, q_fan, p_fan)
            p_total += p_interp
        return {"Q": q_common, "P": p_total}

    # Parallel: sum airflows at the same P
    # Common P range: intersection of all fans' P ranges
    p_min = max(c[1].min() for c in curves)
    p_max = min(c[1].max() for c in curves)
    if p_min >= p_max:
        raise ValueError("Fan curves have no overlapping pressure range for parallel combination.")
    p_common = np.linspace(p_min, p_max, n_points)
    q_total = np.zeros(n_points, dtype=float)
    for q_fan, p_fan in curves:
        # For interpolation, P must be monotonic. Fan curves typically
        # have P decreasing with Q, so we sort by P ascending.
        sort_idx = np.argsort(p_fan)
        p_sorted = p_fan[sort_idx]
        q_sorted = q_fan[sort_idx]
        q_interp = np.interp(p_common, p_sorted, q_sorted)
        q_total += q_interp
    return {"Q": q_total, "P": p_common}
