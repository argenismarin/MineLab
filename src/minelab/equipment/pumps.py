"""Pump sizing and hydraulic calculations for mining dewatering and slurry.

This module provides functions for pump head, power, friction losses,
specific speed, slurry pump derating, and NPSH calculations.

References
----------
.. [1] Karassik, I.J. et al. (2008). *Pump Handbook*, 4th ed. McGraw-Hill.
.. [2] Warman International (2010). *Slurry Pumping Manual*.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Total Dynamic Head
# ---------------------------------------------------------------------------


def pump_head(
    static_head: float,
    velocity_head: float,
    friction_head: float,
) -> float:
    """Calculate Total Dynamic Head (TDH).

    .. math::

        TDH = h_s + h_v + h_f

    Parameters
    ----------
    static_head : float
        Static (elevation) head in metres.
    velocity_head : float
        Velocity head in metres.  Must be non-negative.
    friction_head : float
        Friction head losses in metres.  Must be non-negative.

    Returns
    -------
    float
        Total dynamic head in metres.

    Examples
    --------
    >>> pump_head(30.0, 1.5, 5.0)
    36.5

    References
    ----------
    .. [1] Karassik et al. (2008), Ch. 8.
    """
    validate_non_negative(velocity_head, "velocity_head")
    validate_non_negative(friction_head, "friction_head")

    return float(static_head + velocity_head + friction_head)


# ---------------------------------------------------------------------------
# Pump Power
# ---------------------------------------------------------------------------


def pump_power(
    flow_rate_m3s: float,
    head_m: float,
    efficiency: float,
    fluid_density: float = 1000.0,
) -> float:
    """Calculate pump shaft power.

    .. math::

        P = \\frac{\\rho \\, g \\, Q \\, H}{\\eta \\times 1000}

    Parameters
    ----------
    flow_rate_m3s : float
        Volumetric flow rate in m^3/s.  Must be positive.
    head_m : float
        Total dynamic head in metres.  Must be positive.
    efficiency : float
        Pump efficiency as a fraction (0--1).  Must be in (0, 1].
    fluid_density : float, optional
        Fluid density in kg/m^3 (default 1000).  Must be positive.

    Returns
    -------
    float
        Required shaft power in kW.

    Examples
    --------
    >>> round(pump_power(0.05, 50.0, 0.75), 2)
    32.7

    References
    ----------
    .. [1] Karassik et al. (2008), Ch. 8.
    """
    validate_positive(flow_rate_m3s, "flow_rate_m3s")
    validate_positive(head_m, "head_m")
    validate_range(efficiency, 0.01, 1.0, "efficiency")
    validate_positive(fluid_density, "fluid_density")

    g = 9.81
    power_kw = fluid_density * g * flow_rate_m3s * head_m / (efficiency * 1000.0)

    return float(power_kw)


# ---------------------------------------------------------------------------
# Darcy-Weisbach Friction
# ---------------------------------------------------------------------------


def darcy_weisbach_friction(
    flow_velocity: float,
    pipe_diameter: float,
    pipe_length: float,
    friction_factor: float,
) -> float:
    """Darcy-Weisbach equation for pipe friction head loss.

    .. math::

        h_f = f \\, \\frac{L}{D} \\, \\frac{v^2}{2 g}

    Parameters
    ----------
    flow_velocity : float
        Flow velocity in m/s.  Must be positive.
    pipe_diameter : float
        Internal pipe diameter in metres.  Must be positive.
    pipe_length : float
        Pipe length in metres.  Must be positive.
    friction_factor : float
        Darcy friction factor (dimensionless).  Must be positive.

    Returns
    -------
    float
        Friction head loss in metres.

    Examples
    --------
    >>> round(darcy_weisbach_friction(2.0, 0.2, 100, 0.025), 2)
    2.55

    References
    ----------
    .. [1] Colebrook, C.F. (1939). J. Inst. Civil Engineers, 11, 133--156.
    """
    validate_positive(flow_velocity, "flow_velocity")
    validate_positive(pipe_diameter, "pipe_diameter")
    validate_positive(pipe_length, "pipe_length")
    validate_positive(friction_factor, "friction_factor")

    g = 9.81
    hf = friction_factor * (pipe_length / pipe_diameter) * (flow_velocity**2) / (2.0 * g)

    return float(hf)


# ---------------------------------------------------------------------------
# Pump Specific Speed
# ---------------------------------------------------------------------------


def pump_specific_speed(
    n_rpm: float,
    q_m3s: float,
    h_m: float,
) -> float:
    """Calculate pump specific speed (SI dimensionless form).

    .. math::

        N_s = n \\, \\frac{Q^{0.5}}{H^{0.75}}

    Parameters
    ----------
    n_rpm : float
        Pump rotational speed in rev/min.  Must be positive.
    q_m3s : float
        Flow rate in m^3/s.  Must be positive.
    h_m : float
        Head per stage in metres.  Must be positive.

    Returns
    -------
    float
        Specific speed (dimensionless in SI units).

    Examples
    --------
    >>> round(pump_specific_speed(1450, 0.1, 30), 1)
    86.4

    References
    ----------
    .. [1] Karassik et al. (2008), Ch. 2.
    """
    validate_positive(n_rpm, "n_rpm")
    validate_positive(q_m3s, "q_m3s")
    validate_positive(h_m, "h_m")

    ns = n_rpm * q_m3s**0.5 / h_m**0.75

    return float(ns)


# ---------------------------------------------------------------------------
# Slurry Pump Factor
# ---------------------------------------------------------------------------


def slurry_pump_factor(
    solid_sg: float,
    liquid_sg: float,
    solids_concentration: float,
) -> dict:
    """Calculate slurry pump correction factors.

    Slurry specific gravity, head ratio, and efficiency derating are
    estimated for centrifugal slurry pumps.

    .. math::

        SG_{slurry} = \\frac{SG_{liquid}}{1 - C_w \\left(
        1 - SG_{liquid} / SG_{solid}\\right)}

    Parameters
    ----------
    solid_sg : float
        Specific gravity of solids.  Must be positive and > liquid_sg.
    liquid_sg : float
        Specific gravity of the carrier liquid (typically ~1.0).
        Must be positive.
    solids_concentration : float
        Solids concentration by weight as a fraction (0--1).
        Must be in [0, 1).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"slurry_sg"`` : float -- Slurry specific gravity.
        - ``"head_ratio"`` : float -- Head derating ratio (HR).
        - ``"efficiency_derating"`` : float -- Efficiency derating factor.

    Examples
    --------
    >>> r = slurry_pump_factor(2.65, 1.0, 0.3)
    >>> round(r["slurry_sg"], 3)
    1.228

    References
    ----------
    .. [1] Warman International (2010). *Slurry Pumping Manual*.
    """
    validate_positive(solid_sg, "solid_sg")
    validate_positive(liquid_sg, "liquid_sg")
    validate_range(solids_concentration, 0.0, 0.99, "solids_concentration")

    if solid_sg <= liquid_sg:
        raise ValueError(
            f"'solid_sg' ({solid_sg}) must be greater than 'liquid_sg' ({liquid_sg})."
        )

    denom = 1.0 - solids_concentration * (1.0 - liquid_sg / solid_sg)
    slurry_sg = liquid_sg / denom

    head_ratio = 1.0 - 0.8 * solids_concentration
    efficiency_derating = 1.0 - 0.5 * solids_concentration

    return {
        "slurry_sg": float(slurry_sg),
        "head_ratio": float(head_ratio),
        "efficiency_derating": float(efficiency_derating),
    }


# ---------------------------------------------------------------------------
# NPSH Available
# ---------------------------------------------------------------------------


def npsh_available(
    atmospheric_pressure_m: float,
    suction_head_m: float,
    vapor_pressure_m: float,
    friction_suction_m: float,
) -> float:
    """Calculate Net Positive Suction Head available (NPSHa).

    .. math::

        NPSH_a = P_{atm} + h_s - P_{vap} - h_f

    For suction lift applications, *suction_head_m* is negative.

    Parameters
    ----------
    atmospheric_pressure_m : float
        Atmospheric pressure head in metres of fluid.  Must be positive.
    suction_head_m : float
        Static suction head in metres (positive for flooded suction,
        negative for suction lift).
    vapor_pressure_m : float
        Vapor pressure of the fluid in metres of head.  Must be
        non-negative.
    friction_suction_m : float
        Friction head loss in the suction piping in metres.  Must be
        non-negative.

    Returns
    -------
    float
        NPSHa in metres.

    Examples
    --------
    >>> round(npsh_available(10.33, 3.0, 0.24, 0.5), 2)
    12.59

    References
    ----------
    .. [1] Karassik et al. (2008), Ch. 14.
    """
    validate_positive(atmospheric_pressure_m, "atmospheric_pressure_m")
    validate_non_negative(vapor_pressure_m, "vapor_pressure_m")
    validate_non_negative(friction_suction_m, "friction_suction_m")

    npsha = atmospheric_pressure_m + suction_head_m - vapor_pressure_m - friction_suction_m

    return float(npsha)
