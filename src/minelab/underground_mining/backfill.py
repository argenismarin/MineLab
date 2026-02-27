"""Backfill engineering for underground mining.

Cemented paste fill strength, arching stress, hydraulic fill transport,
pour scheduling, and backfill volume requirements.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Cemented Paste Fill Strength — Belem & Benzaazoua (2008)
# ---------------------------------------------------------------------------


def cemented_paste_strength(
    cement_content: float,
    cure_days: float,
    water_cement_ratio: float,
) -> float:
    """Estimate UCS of cemented paste backfill.

    Parameters
    ----------
    cement_content : float
        Cement content as a mass fraction (e.g. 0.05 for 5%).
    cure_days : float
        Curing time (days).
    water_cement_ratio : float
        Water-to-cement ratio by mass.

    Returns
    -------
    float
        Estimated unconfined compressive strength (kPa).

    Examples
    --------
    >>> cemented_paste_strength(0.05, 28, 7.0)
    100.0

    References
    ----------
    .. [1] Belem, T. & Benzaazoua, M. (2008). "Design and application
       of underground mine paste backfill technology." Geotechnical and
       Geological Engineering, 26(2), 147-174.
    """
    validate_positive(cement_content, "cement_content")
    validate_positive(cure_days, "cure_days")
    validate_positive(water_cement_ratio, "water_cement_ratio")

    # UCS ~ 14 * Cc * sqrt(t/28) / (w/c)
    ucs = 14.0 * cement_content * np.sqrt(cure_days / 28.0) / water_cement_ratio
    return float(ucs)


# ---------------------------------------------------------------------------
# Arching Stress — Marston (1930)
# ---------------------------------------------------------------------------


def arching_stress(
    fill_height: float,
    fill_width: float,
    cohesion: float,
    friction_angle: float,
    density: float,
) -> dict:
    """Vertical stress at the base of a backfilled stope with arching.

    Parameters
    ----------
    fill_height : float
        Fill height (m).
    fill_width : float
        Stope / fill width (m).
    cohesion : float
        Fill cohesion (kPa).
    friction_angle : float
        Fill friction angle (degrees, 1-89).
    density : float
        Fill bulk density (kg/m^3).

    Returns
    -------
    dict
        Keys: ``"vertical_stress_kpa"`` (kPa with arching),
        ``"arching_ratio"`` (ratio to full overburden stress),
        ``"K_ratio"`` (lateral earth pressure coefficient).

    Examples
    --------
    >>> result = arching_stress(30.0, 6.0, 10.0, 35.0, 2000.0)
    >>> result["arching_ratio"] < 1.0
    True

    References
    ----------
    .. [1] Marston, A. (1930). "The theory of external loads on closed
       conduits in the light of the latest experiments." Bulletin 96,
       Iowa Engineering Experiment Station.
    """
    validate_positive(fill_height, "fill_height")
    validate_positive(fill_width, "fill_width")
    validate_positive(cohesion, "cohesion")
    validate_range(friction_angle, 1, 89, "friction_angle")
    validate_positive(density, "density")

    phi_rad = np.radians(friction_angle)

    # Lateral earth pressure coefficient (Rankine active)
    k_ratio = (1.0 - np.sin(phi_rad)) / (1.0 + np.sin(phi_rad))

    # Unit weight (kN/m3)
    gamma = density * 9.81 / 1000.0

    # Marston's arching factor
    k_tan_phi = k_ratio * np.tan(phi_rad)
    b = fill_width  # stope width
    h = fill_height

    if k_tan_phi > 0:
        # sigma_v = (gamma * B) / (2 * K * tan(phi)) *
        #           (1 - exp(-2 * K * tan(phi) * H / B))
        sigma_v = (gamma * b) / (2.0 * k_tan_phi) * (1.0 - np.exp(-2.0 * k_tan_phi * h / b))
    else:
        sigma_v = gamma * h  # no arching

    # Full overburden stress
    sigma_full = gamma * h
    arching_ratio = sigma_v / sigma_full if sigma_full > 0 else 1.0

    return {
        "vertical_stress_kpa": float(sigma_v),
        "arching_ratio": float(arching_ratio),
        "K_ratio": float(k_ratio),
    }


# ---------------------------------------------------------------------------
# Hydraulic Fill Transport — Durand (1953)
# ---------------------------------------------------------------------------


def hydraulic_fill_transport(
    flow_velocity: float,
    pipe_diameter: float,
    slurry_density: float,
) -> dict:
    """Hydraulic fill pipeline transport analysis.

    Parameters
    ----------
    flow_velocity : float
        Slurry flow velocity (m/s).
    pipe_diameter : float
        Pipe internal diameter (m).
    slurry_density : float
        Slurry density (kg/m^3).

    Returns
    -------
    dict
        Keys: ``"critical_velocity_ms"`` (minimum velocity to avoid
        settling, m/s),
        ``"is_above_critical"`` (bool),
        ``"head_loss_kpa_per_m"`` (friction head loss per metre of
        pipe, kPa/m).

    Examples
    --------
    >>> result = hydraulic_fill_transport(2.5, 0.15, 1600.0)
    >>> result["is_above_critical"]
    True

    References
    ----------
    .. [1] Durand, R. (1953). "Basic relationships of the transportation
       of solids in pipes — experimental research." Proc. Int. Assoc.
       Hydraulic Research, Minneapolis, 89-103.
    """
    validate_positive(flow_velocity, "flow_velocity")
    validate_positive(pipe_diameter, "pipe_diameter")
    validate_positive(slurry_density, "slurry_density")

    g = 9.81
    # Critical deposition velocity (simplified Durand)
    # Vc = 1.8 * sqrt(2 * g * D * (S - 1)) where S = rho_s/rho_w
    s_ratio = slurry_density / 1000.0 - 1.0
    s_ratio = max(s_ratio, 0.001)  # prevent negative sqrt
    v_crit = 1.8 * np.sqrt(2.0 * g * pipe_diameter * s_ratio)

    is_above_critical = flow_velocity > v_crit

    # Approximate Darcy-Weisbach style head loss per metre
    # dP/dL ~ f * rho * v^2 / (2 * D), f ~ 0.02
    f_friction = 0.02
    head_loss = (
        f_friction * slurry_density * flow_velocity**2 / (2.0 * pipe_diameter)
    ) / 1000.0  # Pa/m to kPa/m

    return {
        "critical_velocity_ms": float(v_crit),
        "is_above_critical": bool(is_above_critical),
        "head_loss_kpa_per_m": float(head_loss),
    }


# ---------------------------------------------------------------------------
# Fill Pour Rate Scheduling
# ---------------------------------------------------------------------------


def fill_pour_rate(
    stope_volume: float,
    daily_pour_m3: float,
    cure_time_days: float,
) -> dict:
    """Simple backfill pour schedule.

    Parameters
    ----------
    stope_volume : float
        Stope void volume to fill (m^3).
    daily_pour_m3 : float
        Daily pour capacity (m^3/day).
    cure_time_days : float
        Required cure time after pouring (days).

    Returns
    -------
    dict
        Keys: ``"pour_days"`` (pouring duration),
        ``"total_days"`` (pour + cure),
        ``"effective_fill_rate_m3_per_day"``
        (volume / total_days).

    Examples
    --------
    >>> result = fill_pour_rate(5000.0, 200.0, 14.0)
    >>> result["pour_days"]
    25.0

    References
    ----------
    .. [1] Potvin, Y., Thomas, E.G. & Fourie, A.B. (2005). "Handbook
       on Mine Fill." ACG, Perth.
    """
    validate_positive(stope_volume, "stope_volume")
    validate_positive(daily_pour_m3, "daily_pour_m3")
    validate_positive(cure_time_days, "cure_time_days")

    pour_days = stope_volume / daily_pour_m3
    total_days = pour_days + cure_time_days
    effective_rate = stope_volume / total_days

    return {
        "pour_days": float(pour_days),
        "total_days": float(total_days),
        "effective_fill_rate_m3_per_day": float(effective_rate),
    }


# ---------------------------------------------------------------------------
# Backfill Requirement
# ---------------------------------------------------------------------------


def backfill_requirement(
    ore_volume_extracted: float,
    void_filling_ratio: float,
    fill_density: float,
) -> dict:
    """Calculate backfill volume and mass requirements.

    Parameters
    ----------
    ore_volume_extracted : float
        Extracted ore volume (m^3).
    void_filling_ratio : float
        Fraction of void to be filled (0 to 1).
    fill_density : float
        Placed fill density (t/m^3).

    Returns
    -------
    dict
        Keys: ``"fill_volume_m3"`` (required fill volume),
        ``"fill_mass_tonnes"`` (required fill mass).

    Examples
    --------
    >>> result = backfill_requirement(10000.0, 0.95, 1.8)
    >>> result["fill_volume_m3"]
    9500.0

    References
    ----------
    .. [1] Potvin, Y., Thomas, E.G. & Fourie, A.B. (2005). "Handbook
       on Mine Fill." ACG, Perth.
    """
    validate_positive(ore_volume_extracted, "ore_volume_extracted")
    validate_range(void_filling_ratio, 0, 1, "void_filling_ratio")
    validate_positive(fill_density, "fill_density")

    fill_volume = ore_volume_extracted * void_filling_ratio
    fill_mass = fill_volume * fill_density

    return {
        "fill_volume_m3": float(fill_volume),
        "fill_mass_tonnes": float(fill_mass),
    }
