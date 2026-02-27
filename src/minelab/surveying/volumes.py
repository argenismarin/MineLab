"""Volumetric calculations for mining earthworks and stockpiles.

This module provides functions for computing volumes of excavations,
stockpiles, and cross-sections using standard surveying methods.

References
----------
.. [1] Uren, J. & Price, W.F. (2010). *Surveying for Engineers*, 5th ed.
       Palgrave Macmillan.
.. [2] Schofield, W. & Breach, M. (2007). *Engineering Surveying*, 6th ed.
       Butterworth-Heinemann.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
)

# ---------------------------------------------------------------------------
# Prismatoid Volume (Simpson's Prismoidal Rule)
# ---------------------------------------------------------------------------


def prismatoid_volume(
    a_top: float,
    a_mid: float,
    a_bottom: float,
    height: float,
) -> float:
    """Compute volume of a prismatoid using Simpson's prismoidal rule.

    .. math::

        V = \\frac{h}{6} (A_{top} + 4 A_{mid} + A_{bottom})

    Parameters
    ----------
    a_top : float
        Top cross-sectional area in m^2.  Must be non-negative.
    a_mid : float
        Middle cross-sectional area in m^2.  Must be non-negative.
    a_bottom : float
        Bottom cross-sectional area in m^2.  Must be non-negative.
    height : float
        Height (vertical distance) between top and bottom in metres.
        Must be positive.

    Returns
    -------
    float
        Volume in m^3.

    Examples
    --------
    >>> prismatoid_volume(100, 80, 60, 10)
    800.0

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 17.
    """
    validate_non_negative(a_top, "a_top")
    validate_non_negative(a_mid, "a_mid")
    validate_non_negative(a_bottom, "a_bottom")
    validate_positive(height, "height")

    volume = height / 6.0 * (a_top + 4.0 * a_mid + a_bottom)

    return float(volume)


# ---------------------------------------------------------------------------
# Cone Stockpile Volume
# ---------------------------------------------------------------------------


def cone_stockpile_volume(
    base_radius: float,
    height: float,
    angle_of_repose: float,
) -> dict:
    """Calculate volume and surface area of a conical stockpile.

    .. math::

        V = \\frac{\\pi}{3} r^2 h

        A_{surface} = \\pi r \\sqrt{r^2 + h^2}

    Parameters
    ----------
    base_radius : float
        Radius of the stockpile base in metres.  Must be positive.
    height : float
        Stockpile height in metres.  Must be positive.
    angle_of_repose : float
        Material angle of repose in degrees.  Must be in (0, 90).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"volume_m3"`` : float
        - ``"surface_area_m2"`` : float
        - ``"base_area_m2"`` : float

    Examples
    --------
    >>> r = cone_stockpile_volume(10, 5, 30)
    >>> round(r["volume_m3"], 1)
    523.6

    References
    ----------
    .. [1] Schofield & Breach (2007), Ch. 14.
    """
    validate_positive(base_radius, "base_radius")
    validate_positive(height, "height")
    validate_positive(angle_of_repose, "angle_of_repose")
    if angle_of_repose >= 90.0:
        raise ValueError(f"'angle_of_repose' must be less than 90, got {angle_of_repose}.")

    volume = math.pi / 3.0 * base_radius**2 * height
    slant_height = math.sqrt(base_radius**2 + height**2)
    surface_area = math.pi * base_radius * slant_height
    base_area = math.pi * base_radius**2

    return {
        "volume_m3": float(volume),
        "surface_area_m2": float(surface_area),
        "base_area_m2": float(base_area),
    }


# ---------------------------------------------------------------------------
# Trapezoidal Cross-Section Area
# ---------------------------------------------------------------------------


def trapezoidal_cross_section_area(
    width_bottom: float,
    width_top: float,
    height: float,
) -> float:
    """Compute area of a trapezoidal cross-section.

    .. math::

        A = \\frac{(w_b + w_t)}{2} \\times h

    Parameters
    ----------
    width_bottom : float
        Bottom width in metres.  Must be non-negative.
    width_top : float
        Top width in metres.  Must be non-negative.
    height : float
        Height in metres.  Must be positive.

    Returns
    -------
    float
        Cross-sectional area in m^2.

    Examples
    --------
    >>> trapezoidal_cross_section_area(4, 8, 3)
    18.0

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 17.
    """
    validate_non_negative(width_bottom, "width_bottom")
    validate_non_negative(width_top, "width_top")
    validate_positive(height, "height")

    area = (width_bottom + width_top) / 2.0 * height

    return float(area)


# ---------------------------------------------------------------------------
# End-Area Volume
# ---------------------------------------------------------------------------


def end_area_volume(
    areas: list,
    distances: list,
) -> float:
    """Compute volume by the average end-area method.

    .. math::

        V = \\sum_{i=0}^{n-2} \\frac{A_i + A_{i+1}}{2} \\times d_i

    Parameters
    ----------
    areas : list of float
        Cross-sectional areas in m^2.  Must have at least 2 elements.
        All values must be non-negative.
    distances : list of float
        Distances between consecutive sections in metres.  Length must
        be ``len(areas) - 1``.  All values must be positive.

    Returns
    -------
    float
        Total volume in m^3.

    Raises
    ------
    ValueError
        If list lengths are inconsistent or values are invalid.

    Examples
    --------
    >>> end_area_volume([50, 70, 60], [10, 15])
    1575.0

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 17.
    """
    if len(areas) < 2:
        raise ValueError(f"'areas' must have at least 2 elements, got {len(areas)}.")
    if len(distances) != len(areas) - 1:
        raise ValueError(
            f"'distances' must have {len(areas) - 1} elements "
            f"(len(areas) - 1), got {len(distances)}."
        )

    for i, a in enumerate(areas):
        if a < 0:
            raise ValueError(f"'areas[{i}]' must be non-negative, got {a}.")
    for i, d in enumerate(distances):
        validate_positive(d, f"distances[{i}]")

    volume = 0.0
    for i in range(len(distances)):
        volume += (areas[i] + areas[i + 1]) / 2.0 * distances[i]

    return float(volume)


# ---------------------------------------------------------------------------
# Stockpile Mass
# ---------------------------------------------------------------------------


def stockpile_mass(
    volume: float,
    density: float,
    moisture_percent: float,
) -> dict:
    """Estimate dry and wet mass of a stockpile.

    .. math::

        m_{dry} &= V \\times \\rho

        m_{wet} &= m_{dry} \\times (1 + w/100)

    Parameters
    ----------
    volume : float
        Stockpile volume in m^3.  Must be positive.
    density : float
        Dry bulk density in t/m^3.  Must be positive.
    moisture_percent : float
        Moisture content as a percentage (e.g. 8 for 8 %).
        Must be non-negative.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"dry_mass_tonnes"`` : float
        - ``"wet_mass_tonnes"`` : float
        - ``"moisture_tonnes"`` : float

    Examples
    --------
    >>> r = stockpile_mass(1000, 1.8, 5)
    >>> r["dry_mass_tonnes"]
    1800.0

    References
    ----------
    .. [1] Schofield & Breach (2007), Ch. 14.
    """
    validate_positive(volume, "volume")
    validate_positive(density, "density")
    validate_non_negative(moisture_percent, "moisture_percent")

    dry_mass = volume * density
    wet_mass = dry_mass * (1.0 + moisture_percent / 100.0)
    moisture = wet_mass - dry_mass

    return {
        "dry_mass_tonnes": float(dry_mass),
        "wet_mass_tonnes": float(wet_mass),
        "moisture_tonnes": float(moisture),
    }
