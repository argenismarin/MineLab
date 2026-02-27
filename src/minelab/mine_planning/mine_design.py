"""Open-pit mine geometric design calculations.

Provides functions for pit geometry (inter-ramp and overall slope
angles), haul-road ramp design, and volume/tonnage estimation by
the frustum method.

References
----------
.. [1] Hustrulid, W. A., Kuchta, M., & Martin, R. K. (2013). *Open Pit
       Mine Planning and Design* (3rd ed.). CRC Press.
.. [2] SME Mining Engineering Handbook (3rd ed., 2011). Society for
       Mining, Metallurgy & Exploration.
"""

from __future__ import annotations

import math

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)


def pit_geometry(
    bench_height: float,
    berm_width: float,
    face_angle: float,
) -> dict:
    """Compute inter-ramp and overall slope angles from bench geometry.

    The inter-ramp angle (IRA) accounts for the bench face angle and
    the safety berm between benches:

    .. math::

        \\text{IRA} = \\arctan\\!\\left(
            \\frac{H}{\\frac{H}{\\tan(\\alpha)} + W_b}
        \\right)

    where *H* is bench height, *alpha* is face angle, and *W_b* is
    berm width.

    The overall slope angle (OSA) is approximated as equal to the IRA
    when no ramp is present.  With a ramp the OSA is typically 3-5
    degrees less, but that adjustment is project-specific and not
    included here.

    Parameters
    ----------
    bench_height : float
        Bench height in metres.  Must be positive.
    berm_width : float
        Safety berm (catch bench) width in metres.  Non-negative.
    face_angle : float
        Bench face angle in degrees, in (0, 90).

    Returns
    -------
    dict
        ``"inter_ramp_angle"`` : float
            Inter-ramp angle in degrees.
        ``"overall_slope_angle"`` : float
            Overall slope angle in degrees (equal to IRA when no ramp
            offset is specified).

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    H = 10 m, berm = 8 m, face angle = 75 deg:

    >>> result = pit_geometry(10.0, 8.0, 75.0)
    >>> 49.0 < result["inter_ramp_angle"] < 51.0
    True

    References
    ----------
    .. [1] Hustrulid, W. A., Kuchta, M., & Martin, R. K. (2013).
           *Open Pit Mine Planning and Design*, Ch. 9.
    """
    validate_positive(bench_height, "bench_height")
    validate_non_negative(berm_width, "berm_width")
    validate_range(face_angle, 0.01, 89.99, "face_angle")

    face_rad = math.radians(face_angle)
    horizontal_face = bench_height / math.tan(face_rad)
    total_horizontal = horizontal_face + berm_width

    ira_rad = math.atan(bench_height / total_horizontal)
    ira_deg = math.degrees(ira_rad)

    return {
        "inter_ramp_angle": ira_deg,
        "overall_slope_angle": ira_deg,
    }


def ramp_design(
    width: float,
    gradient: float,
    radius: float,
) -> dict:
    """Compute haul-road ramp design parameters.

    Parameters
    ----------
    width : float
        Ramp running-surface width in metres.  Must be positive.
    gradient : float
        Ramp gradient in percent (e.g. 10 for 10 %).  Must be in
        (0, 20].
    radius : float
        Minimum curve (switchback) radius in metres.  Must be positive.

    Returns
    -------
    dict
        ``"effective_width"`` : float
            Effective ramp width including drainage (1.5 * *width*).
        ``"switchback_length"`` : float
            Minimum straight length before a switchback, taken as
            ``pi * radius``.
        ``"gradient_ratio"`` : str
            Gradient expressed as "1 in X" (e.g. "1 in 10").
        ``"vertical_rise_per_100m"`` : float
            Metres of rise per 100 m of ramp length.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    >>> res = ramp_design(25.0, 10.0, 30.0)
    >>> res["effective_width"]
    37.5
    >>> round(res["switchback_length"], 2)
    94.25

    References
    ----------
    .. [1] SME Mining Engineering Handbook (3rd ed., 2011), Ch. 9.3.
    """
    validate_positive(width, "width")
    validate_range(gradient, 0.01, 20.0, "gradient")
    validate_positive(radius, "radius")

    effective_width = 1.5 * width
    switchback_length = math.pi * radius
    gradient_ratio = f"1 in {round(100.0 / gradient)}"
    vertical_rise = gradient  # gradient % means gradient m per 100 m

    return {
        "effective_width": effective_width,
        "switchback_length": switchback_length,
        "gradient_ratio": gradient_ratio,
        "vertical_rise_per_100m": vertical_rise,
    }


def pit_volume_tonnage(
    bench_areas: list[float],
    bench_height: float,
    density: float,
) -> dict:
    """Estimate pit volume and tonnage using the frustum formula.

    Between two successive bench surfaces with areas *A1* and *A2*,
    the frustum volume is:

    .. math::

        V = \\frac{h}{3}\\,(A_1 + A_2 + \\sqrt{A_1 \\cdot A_2})

    Parameters
    ----------
    bench_areas : list of float
        Horizontal cross-sectional areas (m^2) from the top bench
        downward.  Must have at least 2 elements, each non-negative.
    bench_height : float
        Vertical distance between successive bench surfaces in metres.
        Must be positive.
    density : float
        Rock density in t/m^3.  Must be positive.

    Returns
    -------
    dict
        ``"bench_volumes"`` : list of float
            Volume (m^3) of each bench slice (length = len(areas) - 1).
        ``"total_volume"`` : float
            Sum of all bench volumes (m^3).
        ``"total_tonnage"`` : float
            ``total_volume * density`` (t).

    Raises
    ------
    ValueError
        If areas has fewer than 2 elements, bench_height or density is
        non-positive, or any area is negative.

    Examples
    --------
    >>> res = pit_volume_tonnage([1000, 800, 500], 10.0, 2.7)
    >>> len(res["bench_volumes"])
    2
    >>> res["total_volume"] > 0
    True

    References
    ----------
    .. [1] Hustrulid, W. A., Kuchta, M., & Martin, R. K. (2013).
           *Open Pit Mine Planning and Design*, Ch. 6.
    """
    if len(bench_areas) < 2:
        raise ValueError(f"'bench_areas' must have at least 2 elements, got {len(bench_areas)}.")

    validate_positive(bench_height, "bench_height")
    validate_positive(density, "density")

    areas = np.asarray(bench_areas, dtype=float)
    if np.any(areas < 0):
        raise ValueError("All bench areas must be non-negative.")

    bench_volumes: list[float] = []
    for i in range(len(areas) - 1):
        a1, a2 = areas[i], areas[i + 1]
        vol = (bench_height / 3.0) * (a1 + a2 + math.sqrt(a1 * a2))
        bench_volumes.append(vol)

    total_volume = sum(bench_volumes)
    total_tonnage = total_volume * density

    return {
        "bench_volumes": bench_volumes,
        "total_volume": total_volume,
        "total_tonnage": total_tonnage,
    }
