"""Room and pillar mining design calculations.

Pillar safety factor, extraction geometry, barrier pillars,
critical span, and subsidence estimation.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Pillar Safety Factor — Brady & Brown (2006)
# ---------------------------------------------------------------------------


def pillar_safety_factor(
    pillar_strength_mpa: float,
    pillar_stress_mpa: float,
) -> float:
    """Pillar safety factor as strength-to-stress ratio.

    Parameters
    ----------
    pillar_strength_mpa : float
        Pillar strength (MPa).
    pillar_stress_mpa : float
        Pillar stress (MPa).

    Returns
    -------
    float
        Safety factor (dimensionless).

    Examples
    --------
    >>> pillar_safety_factor(50.0, 25.0)
    2.0

    References
    ----------
    .. [1] Brady, B.H.G. & Brown, E.T. (2006). "Rock Mechanics for
       Underground Mining." 3rd ed., Springer. Ch. 13.
    """
    validate_positive(pillar_strength_mpa, "pillar_strength_mpa")
    validate_positive(pillar_stress_mpa, "pillar_stress_mpa")

    return float(pillar_strength_mpa / pillar_stress_mpa)


# ---------------------------------------------------------------------------
# Room and Pillar Geometry — Bieniawski (1992)
# ---------------------------------------------------------------------------


def room_and_pillar_geometry(
    room_width: float,
    pillar_width: float,
    seam_height: float,
) -> dict:
    """Extraction ratio and pillar geometry for square pillars.

    Parameters
    ----------
    room_width : float
        Room (entry) width (m).
    pillar_width : float
        Pillar width (m).
    seam_height : float
        Seam / ore body height (m).

    Returns
    -------
    dict
        Keys: ``"extraction_ratio"`` (dimensionless),
        ``"pillar_area_m2"`` (m^2),
        ``"w_over_h"`` (pillar width-to-height ratio).

    Examples
    --------
    >>> result = room_and_pillar_geometry(6.0, 6.0, 3.0)
    >>> result["extraction_ratio"]
    0.75

    References
    ----------
    .. [1] Bieniawski, Z.T. (1992). "Design methodology in rock
       engineering." Balkema, Rotterdam.
    """
    validate_positive(room_width, "room_width")
    validate_positive(pillar_width, "pillar_width")
    validate_positive(seam_height, "seam_height")

    # For square pillars in a regular grid:
    # e = 1 - (Wp / (Wr + Wp))^2
    total = room_width + pillar_width
    extraction_ratio = 1.0 - (pillar_width / total) ** 2

    pillar_area = pillar_width**2
    w_over_h = pillar_width / seam_height

    return {
        "extraction_ratio": float(extraction_ratio),
        "pillar_area_m2": float(pillar_area),
        "w_over_h": float(w_over_h),
    }


# ---------------------------------------------------------------------------
# Barrier Pillar Width — Obert & Duvall (1967)
# ---------------------------------------------------------------------------


def barrier_pillar_width(
    span: float,
    depth: float,
    ucs: float,
    safety_factor: float,
) -> float:
    """Minimum barrier pillar width from overburden stress.

    Parameters
    ----------
    span : float
        Total span to be protected (m).
    depth : float
        Mining depth (m).
    ucs : float
        Uniaxial compressive strength of pillar rock (MPa).
    safety_factor : float
        Required safety factor (> 1 recommended).

    Returns
    -------
    float
        Barrier pillar width in metres.

    Examples
    --------
    >>> barrier_pillar_width(50.0, 200.0, 60.0, 2.0)
    5.92...

    References
    ----------
    .. [1] Obert, L. & Duvall, W.I. (1967). "Rock Mechanics and the
       Design of Structures in Rock." Wiley, New York.
    """
    validate_positive(span, "span")
    validate_positive(depth, "depth")
    validate_positive(ucs, "ucs")
    validate_positive(safety_factor, "safety_factor")

    # gamma * g in MPa/m (typical rock 2700 kg/m3)
    gamma_g = 2700.0 * 9.81 / 1e6  # ~0.02649 MPa/m

    w_barrier = np.sqrt(depth * span * gamma_g * safety_factor / ucs)
    return float(w_barrier)


# ---------------------------------------------------------------------------
# Critical Span — Lang (1994)
# ---------------------------------------------------------------------------


def critical_span(
    rmr: float,
    depth: float,
    k_ratio: float,
) -> dict:
    """Estimate critical unsupported span from RMR.

    Parameters
    ----------
    rmr : float
        Rock Mass Rating (0-100).
    depth : float
        Mining depth (m).
    k_ratio : float
        Horizontal-to-vertical stress ratio (> 0).

    Returns
    -------
    dict
        Keys: ``"critical_span_m"`` (m),
        ``"stability_class"`` (``"stable"``, ``"marginal"``,
        or ``"unstable"``),
        ``"depth_factor"`` (stress reduction factor).

    Examples
    --------
    >>> result = critical_span(60.0, 200.0, 1.5)
    >>> result["critical_span_m"] > 0
    True

    References
    ----------
    .. [1] Lang, B. (1994). "Span design for entry-type excavations."
       MSc thesis, University of British Columbia.
    """
    validate_range(rmr, 0, 100, "rmr")
    validate_positive(depth, "depth")
    validate_positive(k_ratio, "k_ratio")

    # Empirical critical span from RMR
    cs = 2.0 * 10.0 ** ((rmr - 20.0) / 10.0)

    # Depth / stress adjustment factor
    depth_factor = 1.0 / (1.0 + 0.001 * depth * k_ratio)
    cs *= depth_factor

    # Classification
    if cs > 15.0:
        stability_class = "stable"
    elif cs > 5.0:
        stability_class = "marginal"
    else:
        stability_class = "unstable"

    return {
        "critical_span_m": float(cs),
        "stability_class": stability_class,
        "depth_factor": float(depth_factor),
    }


# ---------------------------------------------------------------------------
# Subsidence Angle — Kratzsch (1983)
# ---------------------------------------------------------------------------


def subsidence_angle(
    overburden_depth: float,
    seam_thickness: float,
    seam_dip: float,
) -> dict:
    """Subsidence trough parameters from longwall / room-and-pillar.

    Parameters
    ----------
    overburden_depth : float
        Depth of overburden above the seam (m).
    seam_thickness : float
        Extracted seam thickness (m).
    seam_dip : float
        Seam dip angle (degrees, 0-90).

    Returns
    -------
    dict
        Keys: ``"angle_of_draw_deg"`` (degrees),
        ``"trough_width_m"`` (surface trough width, m),
        ``"max_subsidence_m"`` (maximum surface subsidence, m).

    Examples
    --------
    >>> result = subsidence_angle(100.0, 3.0, 0.0)
    >>> result["angle_of_draw_deg"]
    35.0

    References
    ----------
    .. [1] Kratzsch, H. (1983). "Mining Subsidence Engineering."
       Springer-Verlag, Berlin.
    """
    validate_positive(overburden_depth, "overburden_depth")
    validate_positive(seam_thickness, "seam_thickness")
    validate_range(seam_dip, 0, 90, "seam_dip")

    # Empirical angle of draw
    angle_of_draw = 35.0 + seam_dip / 3.0

    draw_rad = np.radians(angle_of_draw)
    trough_width = 2.0 * overburden_depth * np.tan(draw_rad)

    # Maximum subsidence (typically 0.9 * seam thickness for longwall)
    max_subsidence = seam_thickness * 0.9

    return {
        "angle_of_draw_deg": float(angle_of_draw),
        "trough_width_m": float(trough_width),
        "max_subsidence_m": float(max_subsidence),
    }
