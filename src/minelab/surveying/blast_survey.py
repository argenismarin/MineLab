"""Blast survey and post-blast measurement tools for mining operations.

This module provides functions for measuring drill deviation, blast
movement vectors, muckpile swell, and excavation dig rates.

References
----------
.. [1] Jimeno, C.L., Jimeno, E.L. & Carcedo, F.J.A. (1995). *Drilling
       and Blasting of Rocks*. A.A. Balkema.
.. [2] Persson, P.A., Holmberg, R. & Lee, J. (1994). *Rock Blasting and
       Explosives Engineering*. CRC Press.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_positive,
)

# ---------------------------------------------------------------------------
# Drill Deviation
# ---------------------------------------------------------------------------


def drill_deviation(
    collar_x: float,
    collar_y: float,
    toe_x: float,
    toe_y: float,
    design_azimuth: float,
    design_dip: float,
) -> dict:
    """Measure drillhole deviation from design.

    Computes the actual azimuth from the horizontal projection of the
    collar-to-toe vector and compares it against the design azimuth.

    Parameters
    ----------
    collar_x : float
        Collar X (easting) coordinate.
    collar_y : float
        Collar Y (northing) coordinate.
    toe_x : float
        Toe X (easting) coordinate.
    toe_y : float
        Toe Y (northing) coordinate.
    design_azimuth : float
        Design azimuth in degrees from north.
    design_dip : float
        Design dip angle in degrees (for reference only; deviation is
        calculated in the horizontal plane).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"actual_azimuth_deg"`` : float -- Actual azimuth [0, 360).
        - ``"azimuth_deviation_deg"`` : float -- Actual minus design.
        - ``"horizontal_displacement_m"`` : float -- Horizontal offset.

    Examples
    --------
    >>> r = drill_deviation(100, 200, 101, 210, 0, -90)
    >>> round(r["actual_azimuth_deg"], 2)
    5.71
    >>> round(r["horizontal_displacement_m"], 2)
    10.05

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 11.
    """
    dx = toe_x - collar_x
    dy = toe_y - collar_y

    horizontal_dist = math.sqrt(dx**2 + dy**2)

    if horizontal_dist < 1e-12:
        actual_azimuth = 0.0
    else:
        actual_azimuth = math.degrees(math.atan2(dx, dy))
        if actual_azimuth < 0:
            actual_azimuth += 360.0

    azimuth_deviation = actual_azimuth - design_azimuth

    return {
        "actual_azimuth_deg": float(actual_azimuth),
        "azimuth_deviation_deg": float(azimuth_deviation),
        "horizontal_displacement_m": float(horizontal_dist),
    }


# ---------------------------------------------------------------------------
# Blast Movement Vector
# ---------------------------------------------------------------------------


def blast_movement_vector(
    easting_pre: float,
    northing_pre: float,
    easting_post: float,
    northing_post: float,
) -> dict:
    """Calculate blast-induced movement from pre- and post-blast surveys.

    Parameters
    ----------
    easting_pre : float
        Pre-blast easting coordinate.
    northing_pre : float
        Pre-blast northing coordinate.
    easting_post : float
        Post-blast easting coordinate.
    northing_post : float
        Post-blast northing coordinate.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"magnitude_m"`` : float -- Movement distance in metres.
        - ``"direction_deg"`` : float -- Direction in degrees [0, 360).
        - ``"dx_m"`` : float -- Easting displacement.
        - ``"dy_m"`` : float -- Northing displacement.

    Examples
    --------
    >>> r = blast_movement_vector(1000, 2000, 1005, 2003)
    >>> round(r["magnitude_m"], 2)
    5.83

    References
    ----------
    .. [1] Persson et al. (1994), Ch. 14.
    """
    dx = easting_post - easting_pre
    dy = northing_post - northing_pre

    magnitude = math.sqrt(dx**2 + dy**2)

    if magnitude < 1e-12:
        direction = 0.0
    else:
        direction = math.degrees(math.atan2(dx, dy))
        if direction < 0:
            direction += 360.0

    return {
        "magnitude_m": float(magnitude),
        "direction_deg": float(direction),
        "dx_m": float(dx),
        "dy_m": float(dy),
    }


# ---------------------------------------------------------------------------
# Muckpile Swell Factor
# ---------------------------------------------------------------------------


def muckpile_swell_factor(
    in_situ_density: float,
    broken_density: float,
) -> float:
    """Calculate the muckpile swell factor.

    .. math::

        SF = \\frac{\\rho_{in\\ situ}}{\\rho_{broken}}

    Parameters
    ----------
    in_situ_density : float
        In-situ rock density in t/m^3.  Must be positive.
    broken_density : float
        Broken (muckpile) density in t/m^3.  Must be positive.

    Returns
    -------
    float
        Swell factor (dimensionless, > 1 for swelling).

    Examples
    --------
    >>> round(muckpile_swell_factor(2.7, 1.8), 2)
    1.5

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 13.
    """
    validate_positive(in_situ_density, "in_situ_density")
    validate_positive(broken_density, "broken_density")

    return float(in_situ_density / broken_density)


# ---------------------------------------------------------------------------
# Dig Rate Survey
# ---------------------------------------------------------------------------


def dig_rate_survey(
    tonnes_mined: float,
    operating_hours: float,
    pass_count: float,
) -> dict:
    """Calculate excavation dig rate metrics from survey data.

    Parameters
    ----------
    tonnes_mined : float
        Total tonnes mined.  Must be positive.
    operating_hours : float
        Operating hours during the survey period.  Must be positive.
    pass_count : float
        Number of excavator passes (bucket loads).  Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"tonnes_per_hour"`` : float
        - ``"tonnes_per_pass"`` : float
        - ``"passes_per_hour"`` : float

    Examples
    --------
    >>> r = dig_rate_survey(5000, 10, 250)
    >>> r["tonnes_per_hour"]
    500.0

    References
    ----------
    .. [1] Jimeno et al. (1995), Ch. 15.
    """
    validate_positive(tonnes_mined, "tonnes_mined")
    validate_positive(operating_hours, "operating_hours")
    validate_positive(pass_count, "pass_count")

    tph = tonnes_mined / operating_hours
    tpp = tonnes_mined / pass_count
    pph = pass_count / operating_hours

    return {
        "tonnes_per_hour": float(tph),
        "tonnes_per_pass": float(tpp),
        "passes_per_hour": float(pph),
    }
