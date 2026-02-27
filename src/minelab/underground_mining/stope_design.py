"""Stope design calculations for underground mining.

Stability graph, hydraulic radius, stope dimensioning, rill angle,
undercut blast geometry, and LHD mucking rate.
"""

from __future__ import annotations

import math

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Mathews Stability Number (Potvin 1988)
# ---------------------------------------------------------------------------


def mathews_stability(
    q_prime: float,
    a: float,
    b: float,
    c: float,
) -> dict:
    """Modified stability number N' using the Potvin (1988) stability graph.

    Parameters
    ----------
    q_prime : float
        Modified Q' value (Barton Q without SRF and Jw adjustments).
    a : float
        Rock stress factor A (0 to 1).
    b : float
        Joint orientation factor B (0.2 to 1).
    c : float
        Gravity adjustment factor C (0 to ~10).

    Returns
    -------
    dict
        Keys: ``"n_prime"`` (modified stability number),
        ``"hydraulic_radius_limit"`` (estimated HR limit in m),
        ``"stability_zone"`` (``"stable"``, ``"transition"``,
        or ``"unstable"``).

    Examples
    --------
    >>> result = mathews_stability(10.0, 0.8, 0.5, 4.0)
    >>> result["n_prime"]
    16.0

    References
    ----------
    .. [1] Potvin, Y. (1988). "Empirical Open Stope Design in Canada."
       PhD thesis, University of British Columbia.
    .. [2] Mathews, K.E. et al. (1981). "Prediction of Stable Excavation
       Spans for Mining at Depths Below 1000 Metres in Hard Rock."
       CANMET Report DSS Serial No. OSQ80-00081.
    """
    validate_positive(q_prime, "q_prime")
    validate_range(a, 0, 1, "a")
    validate_range(b, 0.2, 1, "b")
    validate_positive(c, "c")

    n_prime = q_prime * a * b * c

    # Approximate stability zones from Potvin (1988) chart
    if n_prime > 4.0:
        stability_zone = "stable"
    elif n_prime >= 0.1:
        stability_zone = "transition"
    else:
        stability_zone = "unstable"

    # Empirical HR limit from stability graph (simplified)
    # Based on Potvin (1988): HR_limit ~ 5 * log10(N') + 5
    hr_limit = max(2.0, 5.0 * np.log10(n_prime) + 5.0) if n_prime > 0 else 2.0

    return {
        "n_prime": float(n_prime),
        "hydraulic_radius_limit": float(hr_limit),
        "stability_zone": stability_zone,
    }


# ---------------------------------------------------------------------------
# Hydraulic Radius (Mathews et al. 1981)
# ---------------------------------------------------------------------------


def hydraulic_radius(length: float, width: float) -> float:
    """Hydraulic radius of a rectangular surface.

    Parameters
    ----------
    length : float
        Length of the surface (m).
    width : float
        Width of the surface (m).

    Returns
    -------
    float
        Hydraulic radius HR = Area / Perimeter (m).

    Examples
    --------
    >>> hydraulic_radius(10, 5)
    1.6666666666666667

    References
    ----------
    .. [1] Mathews, K.E. et al. (1981). "Prediction of Stable Excavation
       Spans for Mining at Depths Below 1000 Metres in Hard Rock."
       CANMET Report DSS Serial No. OSQ80-00081.
    """
    validate_positive(length, "length")
    validate_positive(width, "width")

    area = length * width
    perimeter = 2.0 * (length + width)
    return float(area / perimeter)


# ---------------------------------------------------------------------------
# Stope Dimensions from HR Constraint
# ---------------------------------------------------------------------------


def stope_dimensions(
    ore_width: float,
    dip: float,
    height: float,
    hr_limit: float,
) -> dict:
    """Calculate maximum strike length from hydraulic radius constraint.

    Solves HR = (strike * height) / (2 * (strike + height)) <= hr_limit
    for the maximum allowable strike length.

    Parameters
    ----------
    ore_width : float
        Ore body width (m).
    dip : float
        Ore body dip angle (degrees, 0-90).
    height : float
        Stope height (m).
    hr_limit : float
        Maximum allowable hydraulic radius (m).

    Returns
    -------
    dict
        Keys: ``"max_strike_length"`` (m),
        ``"actual_hr"`` (m), ``"stope_volume"`` (m^3).

    Examples
    --------
    >>> result = stope_dimensions(5.0, 70.0, 30.0, 8.0)
    >>> result["max_strike_length"] > 0
    True

    References
    ----------
    .. [1] Mathews, K.E. et al. (1981). "Prediction of Stable Excavation
       Spans for Mining at Depths Below 1000 Metres in Hard Rock."
       CANMET Report DSS Serial No. OSQ80-00081.
    """
    validate_positive(ore_width, "ore_width")
    validate_range(dip, 0, 90, "dip")
    validate_positive(height, "height")
    validate_positive(hr_limit, "hr_limit")

    # HR = (S * H) / (2*(S + H)) = hr_limit
    # S * H = 2 * hr_limit * (S + H)
    # S * H - 2 * hr_limit * S = 2 * hr_limit * H
    # S * (H - 2 * hr_limit) = 2 * hr_limit * H
    denominator = height - 2.0 * hr_limit
    max_strike = 1000.0 if denominator <= 0 else (2.0 * hr_limit * height) / denominator

    actual_hr = (max_strike * height) / (2.0 * (max_strike + height))

    stope_volume = max_strike * height * ore_width

    return {
        "max_strike_length": float(max_strike),
        "actual_hr": float(actual_hr),
        "stope_volume": float(stope_volume),
    }


# ---------------------------------------------------------------------------
# Rill Angle
# ---------------------------------------------------------------------------


def rill_angle(repose_angle: float, dip: float) -> float:
    """Effective rill angle for drawpoints in inclined ore bodies.

    Parameters
    ----------
    repose_angle : float
        Angle of repose of broken ore (degrees, 0-90).
    dip : float
        Ore body dip angle (degrees, 0-90).

    Returns
    -------
    float
        Effective rill angle in degrees.

    Examples
    --------
    >>> rill_angle(37.0, 60.0)  # doctest: +SKIP
    33.21...

    References
    ----------
    .. [1] Hamrin, H. (2001). "Underground Mining Methods: Engineering
       Fundamentals and International Case Studies." SME.
    """
    validate_range(repose_angle, 0, 90, "repose_angle")
    validate_range(dip, 0, 90, "dip")

    repose_rad = np.radians(repose_angle)
    dip_rad = np.radians(dip)

    # Effective rill angle = atan(tan(repose) * sin(dip))
    rill_rad = np.arctan(np.tan(repose_rad) * np.sin(dip_rad))
    return float(np.degrees(rill_rad))


# ---------------------------------------------------------------------------
# Undercut Design
# ---------------------------------------------------------------------------


def undercut_design(
    ore_width: float,
    blast_hole_diam: float,
    powder_factor: float,
) -> dict:
    """Ring blast geometry for undercut design.

    Parameters
    ----------
    ore_width : float
        Width of the ore body (m).
    blast_hole_diam : float
        Blast hole diameter (m).
    powder_factor : float
        Powder factor (kg/m^3).

    Returns
    -------
    dict
        Keys: ``"toe_spacing"`` (m), ``"burden"`` (m),
        ``"holes_per_ring"`` (int),
        ``"explosive_per_ring_kg"`` (kg per unit-height ring).

    Examples
    --------
    >>> result = undercut_design(10.0, 0.089, 0.5)
    >>> result["burden"] > 0
    True

    References
    ----------
    .. [1] Hustrulid, W.A. & Bullock, R.L. (2001). "Underground Mining
       Methods: Engineering Fundamentals and International Case
       Studies." SME.
    """
    validate_positive(ore_width, "ore_width")
    validate_positive(blast_hole_diam, "blast_hole_diam")
    validate_positive(powder_factor, "powder_factor")

    # Empirical rules for ring blasting
    toe_spacing = 28.0 * blast_hole_diam
    burden = 25.0 * blast_hole_diam

    holes_per_ring = max(1, math.ceil(ore_width / toe_spacing))

    # Explosive per ring for unit height (1 m slice)
    explosive_per_ring_kg = burden * ore_width * 1.0 * powder_factor

    return {
        "toe_spacing": float(toe_spacing),
        "burden": float(burden),
        "holes_per_ring": int(holes_per_ring),
        "explosive_per_ring_kg": float(explosive_per_ring_kg),
    }


# ---------------------------------------------------------------------------
# Mucking Rate
# ---------------------------------------------------------------------------


def mucking_rate(
    bucket_capacity: float,
    fill_factor: float,
    cycle_time_min: float,
    density: float,
) -> float:
    """LHD (Load-Haul-Dump) mucking productivity.

    Parameters
    ----------
    bucket_capacity : float
        Nominal bucket capacity (m^3).
    fill_factor : float
        Bucket fill factor (0 to 1).
    cycle_time_min : float
        Round-trip cycle time (minutes).
    density : float
        Broken ore density (t/m^3).

    Returns
    -------
    float
        Production rate in tonnes per hour (t/h).

    Examples
    --------
    >>> mucking_rate(6.0, 0.85, 5.0, 2.7)
    165.24

    References
    ----------
    .. [1] Caterpillar (2017). "Caterpillar Performance Handbook."
       47th ed. Underground LHD productivity estimation.
    """
    validate_positive(bucket_capacity, "bucket_capacity")
    validate_range(fill_factor, 0, 1, "fill_factor")
    validate_positive(cycle_time_min, "cycle_time_min")
    validate_positive(density, "density")

    # tonnes per cycle
    tonnes_per_cycle = bucket_capacity * fill_factor * density
    # cycles per hour
    cycles_per_hour = 60.0 / cycle_time_min
    # productivity
    rate = tonnes_per_cycle * cycles_per_hour
    return float(rate)
