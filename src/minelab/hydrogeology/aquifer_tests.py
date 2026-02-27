"""Aquifer testing functions for groundwater characterisation.

This module implements classical well-test analysis methods used in
hydrogeological investigations for mining projects, including the Theis
solution, Cooper-Jacob approximation, and slug-test transmissivity.

References
----------
.. [1] Theis, C.V. (1935). The relation between the lowering of the
       piezometric surface and the rate and duration of discharge of a
       well using ground-water storage. *Trans. Amer. Geophys. Union*,
       16(2), 519--524.
.. [2] Cooper, H.H. & Jacob, C.E. (1946). A generalized graphical method
       for evaluating formation constants and summarizing well-field
       history. *Trans. Amer. Geophys. Union*, 27(4), 526--534.
.. [3] Bouwer, H. & Rice, R.C. (1976). A slug test for determining
       hydraulic conductivity of unconfined aquifers with completely or
       partially penetrating wells. *Water Resources Res.*, 12(3),
       423--428.
.. [4] Todd, D.K. & Mays, L.W. (2005). *Groundwater Hydrology*, 3rd ed.
       Wiley.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Well function W(u) via series expansion (scipy-free)
# ---------------------------------------------------------------------------


def _well_function(u: float) -> float:
    """Evaluate the Theis well function W(u) using a series expansion.

    W(u) = -gamma - ln(u) + sum_{k=1}^{N} (-1)^{k+1} * u^k / (k * k!)

    where gamma = 0.5772156649... (Euler-Mascheroni constant).

    Parameters
    ----------
    u : float
        Dimensionless argument (must be > 0).

    Returns
    -------
    float
        Value of W(u).
    """
    gamma = 0.5772156649015329
    result = -gamma - math.log(u)
    for k in range(1, 101):
        term = ((-1.0) ** (k + 1)) * (u**k) / (k * math.factorial(k))
        result += term
    return result


# ---------------------------------------------------------------------------
# Theis Drawdown
# ---------------------------------------------------------------------------


def theis_drawdown(
    Q: float,  # noqa: N803
    T: float,  # noqa: N803
    S: float,  # noqa: N803
    r: float,
    t: float,
) -> float:
    """Compute drawdown using the Theis (1935) well function.

    The Theis equation gives transient drawdown in a confined aquifer:

        u = r^2 * S / (4 * T * t)
        s = Q / (4 * pi * T) * W(u)

    where W(u) is the well function evaluated by series expansion.

    Parameters
    ----------
    Q : float
        Pumping rate in m3/day.  Must be > 0.
    T : float
        Transmissivity in m2/day.  Must be > 0.
    S : float
        Storativity (dimensionless).  Must be > 0.
    r : float
        Radial distance from the pumping well in metres.  Must be > 0.
    t : float
        Time since pumping started in days.  Must be > 0.

    Returns
    -------
    float
        Drawdown in metres.

    References
    ----------
    .. [1] Theis, C.V. (1935). The relation between the lowering of the
           piezometric surface and the rate and duration of discharge of
           a well using ground-water storage. *Trans. AGU*, 16, 519--524.
    """
    validate_positive(Q, "Q")
    validate_positive(T, "T")
    validate_positive(S, "S")
    validate_positive(r, "r")
    validate_positive(t, "t")

    u = r**2 * S / (4.0 * T * t)
    w_u = _well_function(u)
    s = Q / (4.0 * math.pi * T) * w_u
    return float(s)


# ---------------------------------------------------------------------------
# Cooper-Jacob Drawdown
# ---------------------------------------------------------------------------


def cooper_jacob_drawdown(
    Q: float,  # noqa: N803
    T: float,  # noqa: N803
    S: float,  # noqa: N803
    r: float,
    t: float,
) -> float:
    """Cooper-Jacob (1946) log approximation for drawdown.

    Valid when u = r^2 S / (4 T t) < 0.05.  The drawdown is:

        s = (2.3 * Q) / (4 * pi * T) * log10(2.25 * T * t / (r^2 * S))

    Parameters
    ----------
    Q : float
        Pumping rate in m3/day.  Must be > 0.
    T : float
        Transmissivity in m2/day.  Must be > 0.
    S : float
        Storativity (dimensionless).  Must be > 0.
    r : float
        Radial distance from the pumping well in metres.  Must be > 0.
    t : float
        Time since pumping started in days.  Must be > 0.

    Returns
    -------
    float
        Drawdown in metres.

    Raises
    ------
    ValueError
        If any parameter is non-positive.

    References
    ----------
    .. [1] Cooper, H.H. & Jacob, C.E. (1946). A generalized graphical
           method for evaluating formation constants. *Trans. AGU*,
           27(4), 526--534.
    """
    validate_positive(Q, "Q")
    validate_positive(T, "T")
    validate_positive(S, "S")
    validate_positive(r, "r")
    validate_positive(t, "t")

    argument = 2.25 * T * t / (r**2 * S)
    s = (2.3 * Q) / (4.0 * math.pi * T) * math.log10(argument)
    return float(s)


# ---------------------------------------------------------------------------
# Theis Recovery
# ---------------------------------------------------------------------------


def theis_recovery(
    Q: float,  # noqa: N803
    T: float,  # noqa: N803
    t_pump: float,
    t_since_stop: float,
) -> float:
    """Compute residual drawdown after pump stops (Theis recovery method).

    s' = (2.3 * Q) / (4 * pi * T) * log10((t_pump + t_since_stop)
         / t_since_stop)

    Parameters
    ----------
    Q : float
        Pumping rate that was applied, in m3/day.  Must be > 0.
    T : float
        Transmissivity in m2/day.  Must be > 0.
    t_pump : float
        Duration of pumping in days.  Must be > 0.
    t_since_stop : float
        Time elapsed since the pump was stopped, in days.  Must be > 0.

    Returns
    -------
    float
        Residual drawdown in metres.

    References
    ----------
    .. [1] Theis, C.V. (1935). Recovery method for transmissivity
           determination. In Todd & Mays (2005), *Groundwater Hydrology*,
           3rd ed., Wiley.
    """
    validate_positive(Q, "Q")
    validate_positive(T, "T")
    validate_positive(t_pump, "t_pump")
    validate_positive(t_since_stop, "t_since_stop")

    ratio = (t_pump + t_since_stop) / t_since_stop
    s_prime = (2.3 * Q) / (4.0 * math.pi * T) * math.log10(ratio)
    return float(s_prime)


# ---------------------------------------------------------------------------
# Transmissivity from Slug Test
# ---------------------------------------------------------------------------


def transmissivity_from_slug(
    r_casing: float,
    r_screen: float,
    water_table_depth: float,
    L_screen: float,  # noqa: N803
    slug_volume: float,
) -> float:
    """Estimate transmissivity from a Bouwer-Rice (1976) slug test.

    A simplified approach computes hydraulic conductivity as:

        K = slug_volume / (pi * r_screen^2 * L_screen * water_table_depth)

    and transmissivity as T = K * L_screen.

    Parameters
    ----------
    r_casing : float
        Casing radius in metres.  Must be > 0.
    r_screen : float
        Screen radius in metres.  Must be > 0.
    water_table_depth : float
        Depth to the water table in metres.  Must be > 0.
    L_screen : float
        Screen length in metres.  Must be > 0.
    slug_volume : float
        Volume of the slug in m3.  Must be > 0.

    Returns
    -------
    float
        Estimated transmissivity in m2/day.

    References
    ----------
    .. [1] Bouwer, H. & Rice, R.C. (1976). A slug test for determining
           hydraulic conductivity of unconfined aquifers. *Water Resour.
           Res.*, 12(3), 423--428.
    """
    validate_positive(r_casing, "r_casing")
    validate_positive(r_screen, "r_screen")
    validate_positive(water_table_depth, "water_table_depth")
    validate_positive(L_screen, "L_screen")
    validate_positive(slug_volume, "slug_volume")

    k = slug_volume / (math.pi * r_screen**2 * L_screen * water_table_depth)
    t_val = k * L_screen
    return float(t_val)


# ---------------------------------------------------------------------------
# Specific Capacity
# ---------------------------------------------------------------------------


def specific_capacity(Q: float, drawdown: float) -> float:  # noqa: N803
    """Compute specific capacity of a well.

    SC = Q / drawdown

    Parameters
    ----------
    Q : float
        Pumping rate in m3/day.  Must be > 0.
    drawdown : float
        Observed drawdown in metres.  Must be > 0.

    Returns
    -------
    float
        Specific capacity in m2/day.

    References
    ----------
    .. [1] Todd, D.K. & Mays, L.W. (2005). *Groundwater Hydrology*,
           3rd ed. Wiley. Sec. 4.4.
    """
    validate_positive(Q, "Q")
    validate_positive(drawdown, "drawdown")

    return float(Q / drawdown)


# ---------------------------------------------------------------------------
# Aquifer Hydraulic Conductivity
# ---------------------------------------------------------------------------


def aquifer_hydraulic_conductivity(
    transmissivity: float,
    aquifer_thickness: float,
) -> float:
    """Compute hydraulic conductivity from transmissivity and thickness.

    K = T / b

    Parameters
    ----------
    transmissivity : float
        Aquifer transmissivity in m2/day.  Must be > 0.
    aquifer_thickness : float
        Saturated thickness in metres.  Must be > 0.

    Returns
    -------
    float
        Hydraulic conductivity in m/day.

    References
    ----------
    .. [1] Todd, D.K. & Mays, L.W. (2005). *Groundwater Hydrology*,
           3rd ed. Wiley. Sec. 3.2.
    """
    validate_positive(transmissivity, "transmissivity")
    validate_positive(aquifer_thickness, "aquifer_thickness")

    return float(transmissivity / aquifer_thickness)
