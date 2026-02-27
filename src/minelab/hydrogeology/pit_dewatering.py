"""Pit dewatering calculations for open-pit and underground mines.

This module provides functions for estimating pit inflow, dewatering well
design, and related hydraulic calculations used in mine water management.

References
----------
.. [1] Powers, J.P. et al. (2007). *Construction Dewatering and Groundwater
       Control*, 3rd ed. Wiley.
.. [2] Thiem, G. (1906). *Hydrologische Methoden*. Gebhardt, Leipzig.
.. [3] Sichardt, W. (1928). *Das Fassungsvermoegen von Rohrbrunnen und
       seine Bedeutung fuer die Grundwasserabsenkung*. Springer.
.. [4] Toth, J. (1963). A theoretical analysis of groundwater flow in
       small drainage basins. *J. Geophys. Res.*, 68(16), 4795--4812.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Darcy Pit Inflow
# ---------------------------------------------------------------------------


def darcy_pit_inflow(
    K: float,  # noqa: N803
    hydraulic_gradient: float,
    area: float,
) -> float:
    """Estimate pit inflow using Darcy's law.

    Q = K * i * A

    Parameters
    ----------
    K : float
        Hydraulic conductivity in m/day.  Must be > 0.
    hydraulic_gradient : float
        Hydraulic gradient (dimensionless).  Must be > 0.
    area : float
        Cross-sectional seepage area in m2.  Must be > 0.

    Returns
    -------
    float
        Inflow rate in m3/day.

    References
    ----------
    .. [1] Powers, J.P. et al. (2007). *Construction Dewatering and
           Groundwater Control*, 3rd ed. Wiley, Ch. 3.
    """
    validate_positive(K, "K")
    validate_positive(hydraulic_gradient, "hydraulic_gradient")
    validate_positive(area, "area")

    return float(K * hydraulic_gradient * area)


# ---------------------------------------------------------------------------
# Toth Seepage
# ---------------------------------------------------------------------------


def toth_seepage(
    K: float,  # noqa: N803
    head_diff: float,
    pit_depth: float,
    pit_area: float,
) -> float:
    """Estimate regional flow seepage into a pit (Toth, 1963).

    Q = K * head_diff * sqrt(pit_area) / pit_depth

    Parameters
    ----------
    K : float
        Hydraulic conductivity in m/day.  Must be > 0.
    head_diff : float
        Hydraulic head difference in metres.  Must be > 0.
    pit_depth : float
        Depth of the pit in metres.  Must be > 0.
    pit_area : float
        Plan-view area of the pit in m2.  Must be > 0.

    Returns
    -------
    float
        Estimated seepage rate in m3/day.

    References
    ----------
    .. [1] Toth, J. (1963). A theoretical analysis of groundwater flow
           in small drainage basins. *J. Geophys. Res.*, 68(16),
           4795--4812.
    """
    validate_positive(K, "K")
    validate_positive(head_diff, "head_diff")
    validate_positive(pit_depth, "pit_depth")
    validate_positive(pit_area, "pit_area")

    return float(K * head_diff * math.sqrt(pit_area) / pit_depth)


# ---------------------------------------------------------------------------
# Dewatering Well Capacity
# ---------------------------------------------------------------------------


def dewatering_well_capacity(
    K: float,  # noqa: N803
    screen_length: float,
    head_reduction: float,
    r_well: float,
    r_influence: float,
) -> float:
    """Steady-state well yield using the Thiem (1906) equation.

    Q = 2 * pi * K * screen_length * head_reduction
        / ln(r_influence / r_well)

    Parameters
    ----------
    K : float
        Hydraulic conductivity in m/day.  Must be > 0.
    screen_length : float
        Length of the well screen in metres.  Must be > 0.
    head_reduction : float
        Target head reduction (drawdown) in metres.  Must be > 0.
    r_well : float
        Well radius in metres.  Must be > 0.
    r_influence : float
        Radius of influence in metres.  Must be > 0 and > *r_well*.

    Returns
    -------
    float
        Well capacity in m3/day.

    Raises
    ------
    ValueError
        If *r_influence* <= *r_well*.

    References
    ----------
    .. [1] Thiem, G. (1906). *Hydrologische Methoden*. Gebhardt, Leipzig.
    """
    validate_positive(K, "K")
    validate_positive(screen_length, "screen_length")
    validate_positive(head_reduction, "head_reduction")
    validate_positive(r_well, "r_well")
    validate_positive(r_influence, "r_influence")
    if r_influence <= r_well:
        raise ValueError(
            "'r_influence' must be greater than 'r_well', got "
            f"r_influence={r_influence}, r_well={r_well}."
        )

    q = 2.0 * math.pi * K * screen_length * head_reduction / math.log(r_influence / r_well)
    return float(q)


# ---------------------------------------------------------------------------
# Number of Dewatering Wells
# ---------------------------------------------------------------------------


def number_of_dewatering_wells(
    total_Q: float,  # noqa: N803
    well_Q: float,  # noqa: N803
    interference_factor: float,
) -> int:
    """Estimate the number of dewatering wells required.

    Accounts for well interference that reduces effective per-well yield:

        effective_Q = well_Q * (1 - interference_factor)
        n = ceil(total_Q / effective_Q)

    Parameters
    ----------
    total_Q : float
        Total required dewatering rate in m3/day.  Must be > 0.
    well_Q : float
        Capacity of a single well in m3/day.  Must be > 0.
    interference_factor : float
        Fraction of capacity lost to interference, in [0, 1).

    Returns
    -------
    int
        Number of dewatering wells required (rounded up).

    References
    ----------
    .. [1] Powers, J.P. et al. (2007). *Construction Dewatering and
           Groundwater Control*, 3rd ed. Wiley, Ch. 8.
    """
    validate_positive(total_Q, "total_Q")
    validate_positive(well_Q, "well_Q")
    validate_range(interference_factor, 0.0, 0.99, "interference_factor")

    effective_q = well_Q * (1.0 - interference_factor)
    return int(math.ceil(total_Q / effective_q))


# ---------------------------------------------------------------------------
# Dewatering Power
# ---------------------------------------------------------------------------


def dewatering_power(
    Q_total: float,  # noqa: N803
    total_dynamic_head: float,
    pump_efficiency: float,
) -> float:
    """Compute pumping power for dewatering.

    P = rho * g * Q * TDH / efficiency

    where rho = 1000 kg/m3, g = 9.81 m/s2, and Q is converted from
    m3/day to m3/s.

    Parameters
    ----------
    Q_total : float
        Total pumping rate in m3/day.  Must be > 0.
    total_dynamic_head : float
        Total dynamic head in metres.  Must be > 0.
    pump_efficiency : float
        Pump efficiency as a fraction, in (0, 1].

    Returns
    -------
    float
        Required pumping power in kW.

    References
    ----------
    .. [1] Powers, J.P. et al. (2007). *Construction Dewatering and
           Groundwater Control*, 3rd ed. Wiley, Ch. 10.
    """
    validate_positive(Q_total, "Q_total")
    validate_positive(total_dynamic_head, "total_dynamic_head")
    validate_positive(pump_efficiency, "pump_efficiency")
    validate_range(pump_efficiency, 0.01, 1.0, "pump_efficiency")

    rho = 1000.0  # kg/m3
    g = 9.81  # m/s2
    q_m3s = Q_total / 86400.0  # m3/day -> m3/s

    # Power in Watts, convert to kW
    p_watts = rho * g * q_m3s * total_dynamic_head / pump_efficiency
    return float(p_watts / 1000.0)


# ---------------------------------------------------------------------------
# Cone of Depression Radius
# ---------------------------------------------------------------------------


def cone_of_depression_radius(
    K: float,  # noqa: N803
    b: float,
    Q: float,  # noqa: N803
    t: float,
    S: float,  # noqa: N803
) -> float:
    """Estimate the radius of the cone of depression.

    Uses the transient approximation:

        R = 1.5 * sqrt(4 * K * b * t / S)

    where T = K * b (transmissivity).

    Parameters
    ----------
    K : float
        Hydraulic conductivity in m/day.  Must be > 0.
    b : float
        Aquifer saturated thickness in metres.  Must be > 0.
    Q : float
        Pumping rate in m3/day.  Must be > 0.
    t : float
        Time since pumping started in days.  Must be > 0.
    S : float
        Storativity (dimensionless).  Must be > 0.

    Returns
    -------
    float
        Radius of cone of depression in metres.

    References
    ----------
    .. [1] Sichardt, W. (1928). *Das Fassungsvermoegen von
           Rohrbrunnen*. Springer.
    .. [2] Powers, J.P. et al. (2007). *Construction Dewatering and
           Groundwater Control*, 3rd ed. Wiley, Ch. 5.
    """
    validate_positive(K, "K")
    validate_positive(b, "b")
    validate_positive(Q, "Q")
    validate_positive(t, "t")
    validate_positive(S, "S")

    radius = 1.5 * math.sqrt(4.0 * K * b * t / S)
    return float(radius)
