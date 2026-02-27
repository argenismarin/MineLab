"""Tailings storage facility design and characterisation.

This module provides simplified calculations for tailings storage capacity
using frustum geometry and empirical beach-angle estimation from slurry
properties.

References
----------
.. [1] Vick, S.G. (1990). *Planning, Design, and Analysis of Tailings
       Dams*. BiTech Publishers.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# Typical dry density of consolidated tailings (t/m3)
_DEFAULT_DRY_DENSITY = 1.4

# ---------------------------------------------------------------------------
# Tailings Storage Capacity
# ---------------------------------------------------------------------------


def tailings_storage_capacity(
    area: float,
    height: float,
    beach_angle: float,
) -> dict:
    """Estimate tailings storage volume using frustum geometry.

    The tailings impoundment is modelled as a truncated cone (frustum).
    Given a base area *A_base*, the top area is reduced by the inward
    recession caused by the beach angle:

    .. math::

        r_{\\text{base}} = \\sqrt{A_{\\text{base}} / \\pi}

        r_{\\text{top}} = r_{\\text{base}} - h \\times \\tan(\\theta)

        A_{\\text{top}} = \\pi \\times r_{\\text{top}}^2

        V = \\frac{h}{3} \\left( A_{\\text{base}} + A_{\\text{top}}
            + \\sqrt{A_{\\text{base}} \\times A_{\\text{top}}} \\right)

    If the beach angle is steep enough that *r_top* would become negative
    the top radius is clamped to zero (full cone).

    Parameters
    ----------
    area : float
        Base area of the impoundment in m2.  Must be positive.
    height : float
        Design height (lift) of the tailings deposit in metres.
        Must be positive.
    beach_angle : float
        Beach angle in degrees (typically 0.5--5).  Must be in [0, 45].

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"volume"`` : float -- Estimated storage volume in m3.
        - ``"tonnes_capacity"`` : float -- Mass capacity assuming a dry
          density of 1.4 t/m3.
        - ``"area_top"`` : float -- Top surface area in m2.

    Examples
    --------
    >>> result = tailings_storage_capacity(10000, 10, 2)
    >>> result["volume"] > 0
    True

    References
    ----------
    .. [1] Vick (1990), Ch. 8 -- Tailings impoundment design.
    """
    validate_positive(area, "area")
    validate_positive(height, "height")
    validate_range(beach_angle, 0, 45, "beach_angle")

    r_base = math.sqrt(area / math.pi)
    r_top = r_base - height * math.tan(math.radians(beach_angle))
    # Clamp to zero if recession exceeds radius
    r_top = max(0.0, r_top)

    a_top = math.pi * r_top**2
    volume = (height / 3.0) * (area + a_top + math.sqrt(area * a_top))
    tonnes = volume * _DEFAULT_DRY_DENSITY

    return {
        "volume": volume,
        "tonnes_capacity": tonnes,
        "area_top": a_top,
    }


# ---------------------------------------------------------------------------
# Beach Angle Estimation
# ---------------------------------------------------------------------------


def tailings_beach_angle(
    solids_conc: float,
    particle_d50: float,
) -> float:
    """Estimate tailings beach angle from slurry properties.

    An empirical relationship relating solids concentration and median
    particle size to the resulting beach slope:

    .. math::

        \\theta \\approx 0.5 + 3 \\times C_w + 0.01 \\times d_{50}

    where *C_w* is solids concentration by weight (fraction 0--1) and
    *d_{50}* is in micrometres.  Typical beach angles range from 0.5 to
    5 degrees.

    Parameters
    ----------
    solids_conc : float
        Solids concentration by weight, as a fraction (0--1).
    particle_d50 : float
        Median particle diameter in micrometres.  Must be positive.

    Returns
    -------
    float
        Estimated beach angle in degrees.

    Examples
    --------
    >>> round(tailings_beach_angle(0.5, 50), 2)
    2.5

    References
    ----------
    .. [1] Vick (1990), Ch. 6 -- Tailings beach formation.
    """
    validate_range(solids_conc, 0, 1, "solids_conc")
    validate_positive(particle_d50, "particle_d50")
    return 0.5 + 3.0 * solids_conc + 0.01 * particle_d50
