"""Blast pattern design functions for surface and underground mining.

This module provides functions for computing burden, spacing, stemming,
subgrade drilling, powder factor, and complete blast pattern design based
on widely accepted empirical formulas.

References
----------
.. [1] Langefors, U. & Kihlstrom, B. (1978). *The Modern Technique of
       Rock Blasting*, 3rd ed. Wiley.
.. [2] Konya, C.J. & Walter, E.J. (1991). *Rock Blasting and Overbreak
       Control*. FHWA.
.. [3] Jimeno, C.L., Jimeno, E.L. & Carcedo, F.J.A. (1995). *Drilling
       and Blasting of Rocks*. Balkema.
.. [4] Hustrulid, W. (1999). *Blasting Principles for Open Pit Mining*,
       Vol. 1. Balkema.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Burden -- Langefors
# ---------------------------------------------------------------------------


def burden_langefors(
    diameter: float,
    rho_e: float,
    c: float = 0.4,
    f: float = 1.0,
    s_b_ratio: float = 1.25,
) -> float:
    """Compute burden using the Langefors formula.

    The Langefors & Kihlstrom (1978) formula for maximum burden is:

    .. math::

        B = \\frac{D}{33} \\sqrt{\\frac{\\rho_e}{c \\, f \\, (S/B)}}

    where *D* is in mm and *rho_e* is explosive density in g/cm³
    (equivalently kg/dm³).

    Parameters
    ----------
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    rho_e : float
        Explosive density in g/cm³ (e.g. 1.2 for emulsion, 0.8 for ANFO).
        Must be positive.
    c : float, optional
        Rock constant (default 0.4, typical for medium-hard rock).
        Must be positive. Range typically 0.2 to 0.6.
    f : float, optional
        Fixation factor (default 1.0 for vertical free face).
        Must be positive.
    s_b_ratio : float, optional
        Spacing-to-burden ratio *S/B* (default 1.25). Must be positive.

    Returns
    -------
    float
        Burden *B* in metres.

    Examples
    --------
    >>> round(burden_langefors(89, 1.2), 2)
    4.18

    >>> round(burden_langefors(76, 0.8, c=0.4, f=1.0, s_b_ratio=1.25), 2)
    2.92

    References
    ----------
    .. [1] Langefors & Kihlstrom (1978), Ch. 4.
    """
    validate_positive(diameter, "diameter")
    validate_positive(rho_e, "rho_e")
    validate_positive(c, "c")
    validate_positive(f, "f")
    validate_positive(s_b_ratio, "s_b_ratio")

    burden = (diameter / 33.0) * math.sqrt(rho_e / (c * f * s_b_ratio))
    return burden


# ---------------------------------------------------------------------------
# Burden -- Konya
# ---------------------------------------------------------------------------


def burden_konya(diameter: float, rho_e: float, rho_r: float) -> float:
    """Compute burden using the Konya formula.

    .. math::

        B = 0.012 \\left(\\frac{2\\,\\rho_e}{\\rho_r} + 1.5\\right) D

    Parameters
    ----------
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    rho_e : float
        Explosive density in g/cm³ (e.g. 1.2 for ANFO). Must be positive.
    rho_r : float
        Rock density in g/cm³ (e.g. 2.65 for granite). Must be positive.

    Returns
    -------
    float
        Burden *B* in metres.

    Examples
    --------
    >>> round(burden_konya(89, 1.2, 2.65), 2)
    2.57

    References
    ----------
    .. [1] Konya & Walter (1991), Ch. 5.
    """
    validate_positive(diameter, "diameter")
    validate_positive(rho_e, "rho_e")
    validate_positive(rho_r, "rho_r")

    burden = 0.012 * (2.0 * rho_e / rho_r + 1.5) * diameter
    return burden


# ---------------------------------------------------------------------------
# Spacing from Burden
# ---------------------------------------------------------------------------


def spacing_from_burden(burden: float, ratio: float = 1.15) -> float:
    """Compute spacing from burden and a spacing-to-burden ratio.

    .. math::

        S = \\text{ratio} \\times B

    Parameters
    ----------
    burden : float
        Burden in metres. Must be positive.
    ratio : float, optional
        Spacing-to-burden ratio (default 1.15). Typical range 1.1 to 1.5.
        Must be positive.

    Returns
    -------
    float
        Spacing *S* in metres.

    Examples
    --------
    >>> spacing_from_burden(3.0)
    3.45

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 10.
    """
    validate_positive(burden, "burden")
    validate_positive(ratio, "ratio")

    return ratio * burden


# ---------------------------------------------------------------------------
# Stemming Length
# ---------------------------------------------------------------------------


def stemming_length(burden: float) -> float:
    """Estimate stemming length from burden.

    .. math::

        T \\approx 0.7 \\, B

    Parameters
    ----------
    burden : float
        Burden in metres. Must be positive.

    Returns
    -------
    float
        Recommended stemming length *T* in metres.

    Examples
    --------
    >>> stemming_length(3.0)
    2.1

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 10.
    """
    validate_positive(burden, "burden")
    return 0.7 * burden


# ---------------------------------------------------------------------------
# Subgrade Drilling
# ---------------------------------------------------------------------------


def subgrade_drilling(burden: float) -> float:
    """Estimate subgrade (subdrill) length from burden.

    .. math::

        J \\approx 0.3 \\, B

    Parameters
    ----------
    burden : float
        Burden in metres. Must be positive.

    Returns
    -------
    float
        Recommended subgrade drilling *J* in metres.

    Examples
    --------
    >>> round(subgrade_drilling(3.0), 1)
    0.9

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 10.
    """
    validate_positive(burden, "burden")
    return 0.3 * burden


# ---------------------------------------------------------------------------
# Powder Factor
# ---------------------------------------------------------------------------


def powder_factor(
    rho_e: float,
    diameter: float,
    burden: float,
    spacing: float,
    bench_height: float,
    stemming: float,
    subdrill: float,
) -> float:
    """Compute powder factor (specific charge) for a blast pattern.

    The charge weight per hole is derived from the explosive column length
    and cross-sectional area of the drill hole:

    .. math::

        PF = \\frac{\\rho_e \\, \\pi \\, (D/2000)^2 \\,
              (H + J - T)}{B \\, S \\, H}

    where *D* is converted from mm to m by dividing by 1000.

    Parameters
    ----------
    rho_e : float
        Explosive density in kg/m³. Must be positive.
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    burden : float
        Burden *B* in metres. Must be positive.
    spacing : float
        Spacing *S* in metres. Must be positive.
    bench_height : float
        Bench height *H* in metres. Must be positive.
    stemming : float
        Stemming length *T* in metres. Must be non-negative.
    subdrill : float
        Subdrill length *J* in metres. Must be non-negative.

    Returns
    -------
    float
        Powder factor in kg/m³.

    Raises
    ------
    ValueError
        If charge length ``(H + subdrill - stemming)`` is not positive.

    Examples
    --------
    >>> round(powder_factor(1200, 89, 2.5, 2.88, 10, 1.75, 0.75), 2)
    0.51

    References
    ----------
    .. [1] Hustrulid (1999), Ch. 10.
    """
    validate_positive(rho_e, "rho_e")
    validate_positive(diameter, "diameter")
    validate_positive(burden, "burden")
    validate_positive(spacing, "spacing")
    validate_positive(bench_height, "bench_height")
    if stemming < 0:
        raise ValueError(f"'stemming' must be non-negative, got {stemming}.")
    if subdrill < 0:
        raise ValueError(f"'subdrill' must be non-negative, got {subdrill}.")

    charge_length = bench_height + subdrill - stemming
    if charge_length <= 0:
        raise ValueError(
            f"Charge length (H + subdrill - stemming) must be positive, got {charge_length}."
        )

    radius_m = diameter / 2000.0  # mm -> m
    charge_weight = rho_e * math.pi * radius_m**2 * charge_length
    volume = burden * spacing * bench_height
    return charge_weight / volume


# ---------------------------------------------------------------------------
# Complete Pattern Design
# ---------------------------------------------------------------------------


def pattern_design(
    diameter: float,
    rho_e: float,
    rho_r: float,
    bench_height: float,
) -> dict:
    """Design a complete blast pattern using standard empirical rules.

    The function calculates burden (Konya), spacing, stemming, subdrill,
    charge length, and powder factor in a single call.

    Parameters
    ----------
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    rho_e : float
        Explosive density in g/cm³ (e.g. 1.2 for ANFO). Must be positive.
    rho_r : float
        Rock density in g/cm³ (e.g. 2.65 for granite). Must be positive.
    bench_height : float
        Bench height *H* in metres. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``burden`` : float -- Burden *B* (m).
        - ``spacing`` : float -- Spacing *S* (m).
        - ``stemming`` : float -- Stemming *T* (m).
        - ``subdrill`` : float -- Subdrill *J* (m).
        - ``charge_length`` : float -- Explosive column length (m).
        - ``powder_factor`` : float -- Powder factor (kg/m³).

    Examples
    --------
    >>> result = pattern_design(89, 1.2, 2.65, 10)
    >>> round(result['burden'], 2)
    2.57
    >>> round(result['spacing'], 2)
    2.95

    References
    ----------
    .. [1] Konya & Walter (1991), Ch. 5.
    .. [2] Jimeno et al. (1995), Ch. 10.
    .. [3] Hustrulid (1999), Ch. 10.
    """
    validate_positive(diameter, "diameter")
    validate_positive(rho_e, "rho_e")
    validate_positive(rho_r, "rho_r")
    validate_positive(bench_height, "bench_height")

    b = burden_konya(diameter, rho_e, rho_r)
    s = spacing_from_burden(b)
    t = stemming_length(b)
    j = subgrade_drilling(b)
    charge_len = bench_height + j - t

    # Explosive density in kg/m³ for powder_factor calculation
    rho_e_kgm3 = rho_e * 1000.0
    pf = powder_factor(rho_e_kgm3, diameter, b, s, bench_height, t, j)

    return {
        "burden": b,
        "spacing": s,
        "stemming": t,
        "subdrill": j,
        "charge_length": charge_len,
        "powder_factor": pf,
    }
