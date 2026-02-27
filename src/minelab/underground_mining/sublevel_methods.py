"""Sublevel mining methods: caving, stoping, and draw theory.

Sublevel interval calculation, draw ellipsoid geometry, recovery
estimation, ring blast design, and block caving draw rates.
"""

from __future__ import annotations

import math

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Sublevel Interval — Janelid & Kvapil (1966)
# ---------------------------------------------------------------------------


def sublevel_interval(
    ore_dip: float,
    draw_angle: float,
    burden: float,
) -> float:
    """Sublevel interval from ore dip, draw angle, and burden.

    Parameters
    ----------
    ore_dip : float
        Ore body dip angle (degrees, 1-90).
    draw_angle : float
        Draw cone half-angle (degrees, 1-89).
    burden : float
        Ring burden (m).

    Returns
    -------
    float
        Sublevel interval in metres.

    Examples
    --------
    >>> sublevel_interval(70.0, 60.0, 3.0)
    4.870...

    References
    ----------
    .. [1] Janelid, I. & Kvapil, R. (1966). "Sublevel caving."
       Int. J. Rock Mechanics and Mining Sciences, 3(2), 129-153.
    """
    validate_range(ore_dip, 1, 90, "ore_dip")
    validate_range(draw_angle, 1, 89, "draw_angle")
    validate_positive(burden, "burden")

    dip_rad = np.radians(ore_dip)
    draw_rad = np.radians(draw_angle)

    # SI = burden * (1/tan(draw_angle) + 1/tan(dip))
    si = burden * (1.0 / np.tan(draw_rad) + 1.0 / np.tan(dip_rad))
    return float(si)


# ---------------------------------------------------------------------------
# Draw Ellipsoid — Kvapil (1982)
# ---------------------------------------------------------------------------


def draw_ellipsoid(
    height: float,
    draw_angle: float,
) -> dict:
    """Semi-axes and volume of the draw ellipsoid.

    Parameters
    ----------
    height : float
        Draw column height (m).
    draw_angle : float
        Draw cone half-angle at the base (degrees, 1-89).

    Returns
    -------
    dict
        Keys: ``"semi_major_m"`` (vertical half-axis),
        ``"semi_minor_m"`` (horizontal half-axis),
        ``"volume_m3"`` (ellipsoid volume),
        ``"eccentricity"`` (ellipsoid eccentricity).

    Examples
    --------
    >>> result = draw_ellipsoid(30.0, 60.0)
    >>> result["semi_major_m"]
    15.0

    References
    ----------
    .. [1] Kvapil, R. (1982). "The mechanics and design of sublevel
       caving systems." Underground Mining Methods Handbook, SME,
       880-897.
    """
    validate_positive(height, "height")
    validate_range(draw_angle, 1, 89, "draw_angle")

    a = height / 2.0  # vertical semi-axis
    draw_rad = np.radians(draw_angle)
    b = height * np.tan(draw_rad) / 2.0  # horizontal semi-axis

    volume = (4.0 / 3.0) * np.pi * a * b * b

    # Eccentricity of prolate (a > b) or oblate (b > a) spheroid
    eccentricity = np.sqrt(1.0 - (b / a) ** 2) if a >= b else np.sqrt(1.0 - (a / b) ** 2)

    return {
        "semi_major_m": float(a),
        "semi_minor_m": float(b),
        "volume_m3": float(volume),
        "eccentricity": float(eccentricity),
    }


# ---------------------------------------------------------------------------
# Sublevel Recovery — Laubscher (1994)
# ---------------------------------------------------------------------------


def sublevel_recovery(
    draw_height: float,
    sublevel_interval: float,
    ore_density: float,
    waste_density: float,
) -> dict:
    """Ore recovery and dilution estimate for sublevel methods.

    Parameters
    ----------
    draw_height : float
        Effective draw height (m).
    sublevel_interval : float
        Sublevel interval (m).
    ore_density : float
        Ore density (t/m^3).
    waste_density : float
        Waste rock density (t/m^3).

    Returns
    -------
    dict
        Keys: ``"recovery_fraction"`` (0 to 1),
        ``"dilution_fraction"`` (>= 0),
        ``"ore_extracted_factor"`` (effective extraction factor).

    Examples
    --------
    >>> result = sublevel_recovery(25.0, 30.0, 3.0, 2.7)
    >>> 0 < result["recovery_fraction"] <= 1
    True

    References
    ----------
    .. [1] Laubscher, D.H. (1994). "Cave mining — the state of the
       art." Journal of the SAIMM, 94(10), 279-293.
    """
    validate_positive(draw_height, "draw_height")
    validate_positive(sublevel_interval, "sublevel_interval")
    validate_positive(ore_density, "ore_density")
    validate_positive(waste_density, "waste_density")

    draw_ratio = draw_height / sublevel_interval

    # Empirical recovery and dilution
    recovery = min(1.0, draw_ratio * 0.85)
    dilution = max(0.0, (1.0 - draw_ratio) * waste_density / ore_density * 0.3)

    ore_extracted_factor = recovery * (1.0 - dilution)

    return {
        "recovery_fraction": float(recovery),
        "dilution_fraction": float(dilution),
        "ore_extracted_factor": float(ore_extracted_factor),
    }


# ---------------------------------------------------------------------------
# Ring Blast Design — Hustrulid & Bullock (2001)
# ---------------------------------------------------------------------------


def ring_blast_design(
    diameter: float,
    burden: float,
    toe_spacing: float,
) -> dict:
    """Basic ring blast pattern metrics.

    Parameters
    ----------
    diameter : float
        Blast hole diameter (m).
    burden : float
        Ring burden (m).
    toe_spacing : float
        Toe spacing between holes in the ring (m).

    Returns
    -------
    dict
        Keys: ``"area_per_ring_m2"`` (burden x toe_spacing),
        ``"pattern_ratio"`` (toe_spacing / burden),
        ``"drill_metres_per_ring"`` (estimated drill metres).

    Examples
    --------
    >>> result = ring_blast_design(0.089, 2.5, 3.0)
    >>> result["area_per_ring_m2"]
    7.5

    References
    ----------
    .. [1] Hustrulid, W.A. & Bullock, R.L. (2001). "Underground Mining
       Methods: Engineering Fundamentals and International Case
       Studies." SME.
    """
    validate_positive(diameter, "diameter")
    validate_positive(burden, "burden")
    validate_positive(toe_spacing, "toe_spacing")

    area_per_ring = burden * toe_spacing
    pattern_ratio = toe_spacing / burden

    # Approximate drill metres per ring:
    # ring depth ~ burden * 1.1 (for angled holes)
    ring_depth = burden * 1.1
    # Approximate number of holes based on ring height ~ burden * 3
    approx_holes = max(1, math.ceil((burden * 3.0) / toe_spacing))
    drill_metres = approx_holes * ring_depth

    return {
        "area_per_ring_m2": float(area_per_ring),
        "pattern_ratio": float(pattern_ratio),
        "drill_metres_per_ring": float(drill_metres),
    }


# ---------------------------------------------------------------------------
# Block Cave Draw Rate — Laubscher (1994)
# ---------------------------------------------------------------------------


def block_cave_draw_rate(
    column_height: float,
    cave_rate: float,
    footprint_area: float,
    density: float,
) -> dict:
    """Block cave production and draw schedule estimation.

    Parameters
    ----------
    column_height : float
        Ore column height (m).
    cave_rate : float
        Cave propagation / draw rate (m/day).
    footprint_area : float
        Active footprint area (m^2).
    density : float
        Ore density (t/m^3).

    Returns
    -------
    dict
        Keys: ``"total_ore_tonnes"`` (total recoverable ore),
        ``"draw_time_days"`` (time to draw full column),
        ``"daily_production_tonnes"`` (daily production rate).

    Examples
    --------
    >>> result = block_cave_draw_rate(200.0, 0.5, 5000.0, 2.7)
    >>> result["total_ore_tonnes"]
    2700000.0

    References
    ----------
    .. [1] Laubscher, D.H. (1994). "Cave mining — the state of the
       art." Journal of the SAIMM, 94(10), 279-293.
    """
    validate_positive(column_height, "column_height")
    validate_positive(cave_rate, "cave_rate")
    validate_positive(footprint_area, "footprint_area")
    validate_positive(density, "density")

    total_ore = column_height * footprint_area * density
    draw_time = column_height / cave_rate
    daily_production = footprint_area * cave_rate * density

    return {
        "total_ore_tonnes": float(total_ore),
        "draw_time_days": float(draw_time),
        "daily_production_tonnes": float(daily_production),
    }
