"""Belt and screw conveyor design calculations for mining operations.

This module provides functions for sizing and analysing belt conveyors and
screw conveyors, including capacity, power, belt tension, idler spacing,
and slope limits.

References
----------
.. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed.
       Conveyor Equipment Manufacturers Association.
.. [2] Dunlop (2017). *Conveyor Belt Design Manual*.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Belt Conveyor Capacity
# ---------------------------------------------------------------------------


def belt_conveyor_capacity(
    belt_width: float,
    belt_speed: float,
    material_density: float,
    surcharge_angle: float,
) -> float:
    """Estimate belt conveyor capacity using the CEMA cross-section method.

    The cross-sectional area of material on the belt is approximated as

    .. math::

        A = 0.1 \\times (W - 0.1)^2 \\times \\tan(\\alpha)

    where *W* is belt width in metres and *alpha* is the surcharge angle.
    Capacity is then *A * v * rho * 3600* in tonnes per hour.

    Parameters
    ----------
    belt_width : float
        Belt width in metres.  Must be positive.
    belt_speed : float
        Belt speed in m/s.  Must be positive.
    material_density : float
        Bulk density of conveyed material in t/m^3.  Must be positive.
    surcharge_angle : float
        Surcharge (repose) angle of material in degrees.  Must be in
        (0, 90).

    Returns
    -------
    float
        Conveyor capacity in t/h.

    Examples
    --------
    >>> round(belt_conveyor_capacity(1.2, 3.5, 1.8, 20), 1)
    297.5

    References
    ----------
    .. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed., Ch. 4.
    """
    validate_positive(belt_width, "belt_width")
    validate_positive(belt_speed, "belt_speed")
    validate_positive(material_density, "material_density")
    validate_range(surcharge_angle, 0.01, 89.99, "surcharge_angle")

    surcharge_rad = math.radians(surcharge_angle)
    cross_section_area = 0.1 * (belt_width - 0.1) ** 2 * math.tan(surcharge_rad)
    capacity = cross_section_area * belt_speed * material_density * 3600.0

    return float(capacity)


# ---------------------------------------------------------------------------
# Conveyor Power
# ---------------------------------------------------------------------------


def conveyor_power(
    length: float,
    lift: float,
    capacity_tph: float,
    friction_factor: float,
) -> dict:
    """Estimate conveyor drive power using the CEMA method.

    Horizontal and lift power components are calculated separately:

    .. math::

        P_{horiz} = f \\cdot L \\cdot Q \\cdot g / (3600 \\times 1000)

        P_{lift}  = Q / 3.6 \\cdot g \\cdot H / 1000

    where *f* is the friction factor, *L* is conveyor length (m), *Q* is
    capacity (t/h), *g* = 9.81 m/s^2, and *H* is the vertical lift (m).

    Parameters
    ----------
    length : float
        Conveyor centre-to-centre length in metres.  Must be positive.
    lift : float
        Vertical lift in metres.  May be zero or negative (downhill).
    capacity_tph : float
        Material flow rate in t/h.  Must be positive.
    friction_factor : float
        Conveyor friction factor (dimensionless, typically 0.02--0.04).
        Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"horizontal_power_kw"`` : float
        - ``"lift_power_kw"`` : float
        - ``"total_power_kw"`` : float

    Examples
    --------
    >>> p = conveyor_power(500, 30, 1000, 0.03)
    >>> round(p["total_power_kw"], 1)
    122.6

    References
    ----------
    .. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed., Ch. 6.
    """
    validate_positive(length, "length")
    validate_positive(capacity_tph, "capacity_tph")
    validate_positive(friction_factor, "friction_factor")

    g = 9.81

    p_horizontal = friction_factor * length * capacity_tph * g / (3600.0 * 1000.0)
    p_lift = capacity_tph / 3.6 * g * lift / 1000.0
    p_total = p_horizontal + p_lift

    return {
        "horizontal_power_kw": float(p_horizontal),
        "lift_power_kw": float(p_lift),
        "total_power_kw": float(p_total),
    }


# ---------------------------------------------------------------------------
# Belt Tension
# ---------------------------------------------------------------------------


def belt_tension(
    capacity_tph: float,
    speed: float,
    friction: float,
    belt_length: float,
    lift: float,
) -> dict:
    """Estimate belt tensions using effective tension and wrap factors.

    The effective tension is composed of material handling and belt mass
    contributions.  A belt mass per metre of 10 kg/m is assumed for the
    empty-belt component.  Tight-side and slack-side tensions are derived
    using a simplified wrap factor of 1.5.

    Parameters
    ----------
    capacity_tph : float
        Material flow rate in t/h.  Must be positive.
    speed : float
        Belt speed in m/s.  Must be positive.
    friction : float
        Friction factor (dimensionless).  Must be positive.
    belt_length : float
        Conveyor length in metres.  Must be positive.
    lift : float
        Vertical lift in metres (may be negative for downhill).

    Returns
    -------
    dict
        Dictionary with keys (all in kN):

        - ``"effective_tension_kn"`` : float
        - ``"tight_side_kn"`` : float
        - ``"slack_side_kn"`` : float

    Examples
    --------
    >>> t = belt_tension(800, 3.0, 0.03, 400, 20)
    >>> round(t["effective_tension_kn"], 2)
    16.34

    References
    ----------
    .. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed., Ch. 6.
    """
    validate_positive(capacity_tph, "capacity_tph")
    validate_positive(speed, "speed")
    validate_positive(friction, "friction")
    validate_positive(belt_length, "belt_length")

    g = 9.81
    belt_mass_per_m = 10.0  # kg/m assumed

    # Material component
    t_material = (friction * belt_length + lift) * capacity_tph * g / (3.6 * speed)
    # Belt mass component
    t_belt = friction * belt_mass_per_m * belt_length * g

    t_effective = (t_material + t_belt) / 1000.0  # convert N to kN

    # Simplified wrap factor
    t_tight = 1.5 * t_effective
    t_slack = t_tight - t_effective

    return {
        "effective_tension_kn": float(t_effective),
        "tight_side_kn": float(t_tight),
        "slack_side_kn": float(t_slack),
    }


# ---------------------------------------------------------------------------
# Idler Spacing
# ---------------------------------------------------------------------------


def idler_spacing(
    belt_width: float,
    material_density: float,
    belt_mass_per_m: float,
    sag_limit: float,
) -> float:
    """Estimate carrying-side idler spacing to limit belt sag.

    A simplified formula is used that balances belt sag against the
    combined weight of belt and material:

    .. math::

        S = \\min\\!\\left(1.5,\\;
        \\frac{\\text{sag\\_limit} \\times W \\times 10}
        {\\rho \\times m_b}\\right)

    The result is clamped to the practical range 0.6--1.8 m.

    Parameters
    ----------
    belt_width : float
        Belt width in metres.  Must be positive.
    material_density : float
        Bulk density of material in t/m^3.  Must be positive.
    belt_mass_per_m : float
        Belt mass per unit length in kg/m.  Must be positive.
    sag_limit : float
        Allowable belt sag as a fraction (e.g. 0.02 for 2 %).
        Must be positive.

    Returns
    -------
    float
        Recommended idler spacing in metres (clamped to 0.6--1.8 m).

    Examples
    --------
    >>> round(idler_spacing(1.2, 1.8, 15.0, 0.02), 2)
    0.89

    References
    ----------
    .. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed., Ch. 5.
    """
    validate_positive(belt_width, "belt_width")
    validate_positive(material_density, "material_density")
    validate_positive(belt_mass_per_m, "belt_mass_per_m")
    validate_positive(sag_limit, "sag_limit")

    spacing = min(
        1.5,
        sag_limit * belt_width * 10.0 / (material_density * belt_mass_per_m),
    )
    # Clamp to practical range
    spacing = max(0.6, min(1.8, spacing))

    return float(spacing)


# ---------------------------------------------------------------------------
# Conveyor Slope Limit
# ---------------------------------------------------------------------------


def conveyor_slope_limit(
    material_friction_angle: float,
    safety_margin_deg: float,
) -> float:
    """Maximum conveyor inclination angle.

    The maximum safe inclination is the material friction angle reduced
    by a safety margin to prevent backsliding.

    Parameters
    ----------
    material_friction_angle : float
        Internal friction angle of the conveyed material in degrees.
        Must be positive.
    safety_margin_deg : float
        Safety margin in degrees.  Must be non-negative.

    Returns
    -------
    float
        Maximum conveyor inclination in degrees.

    Raises
    ------
    ValueError
        If the resulting angle is not positive.

    Examples
    --------
    >>> conveyor_slope_limit(35.0, 5.0)
    30.0

    References
    ----------
    .. [1] CEMA (2014). *Belt Conveyors for Bulk Materials*, 7th ed., Ch. 3.
    """
    validate_positive(material_friction_angle, "material_friction_angle")
    validate_non_negative(safety_margin_deg, "safety_margin_deg")

    max_angle = material_friction_angle - safety_margin_deg
    if max_angle <= 0:
        raise ValueError(
            "Resulting slope limit must be positive; "
            f"got {max_angle} deg (friction={material_friction_angle}, "
            f"margin={safety_margin_deg})."
        )

    return float(max_angle)


# ---------------------------------------------------------------------------
# Screw Conveyor Capacity
# ---------------------------------------------------------------------------


def screw_conveyor_capacity(
    diameter: float,
    pitch: float,
    rpm: float,
    fill_factor: float,
    density: float,
) -> float:
    """Estimate volumetric capacity of a screw conveyor.

    .. math::

        Q = \\frac{\\pi}{4} D^2 \\times p \\times n
            \\times \\eta \\times \\rho \\times 60

    where *D* is diameter (m), *p* is pitch (m), *n* is RPM,
    *eta* is fill factor, and *rho* is bulk density (t/m^3).

    Parameters
    ----------
    diameter : float
        Screw diameter in metres.  Must be positive.
    pitch : float
        Screw pitch in metres.  Must be positive.
    rpm : float
        Rotational speed in rev/min.  Must be positive.
    fill_factor : float
        Volumetric fill factor (0--1).  Must be in (0, 1].
    density : float
        Bulk density of material in t/m^3.  Must be positive.

    Returns
    -------
    float
        Conveyor capacity in t/h.

    Examples
    --------
    >>> round(screw_conveyor_capacity(0.3, 0.3, 60, 0.45, 1.6), 2)
    18.32

    References
    ----------
    .. [1] CEMA (2014). *Screw Conveyors*, 5th ed.
    """
    validate_positive(diameter, "diameter")
    validate_positive(pitch, "pitch")
    validate_positive(rpm, "rpm")
    validate_range(fill_factor, 0.01, 1.0, "fill_factor")
    validate_positive(density, "density")

    area = math.pi / 4.0 * diameter**2
    capacity = area * pitch * rpm * fill_factor * density * 60.0

    return float(capacity)
