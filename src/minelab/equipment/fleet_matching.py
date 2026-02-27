"""Truck-loader fleet matching and fleet sizing for open-pit operations.

This module implements the Match Factor methodology and optimal fleet sizing
to balance truck and loader operations in surface mining.

References
----------
.. [1] SME Mining Engineering Handbook, 3rd ed. (2011). Society for Mining,
       Metallurgy & Exploration, Ch. 9.5.
.. [2] Hustrulid, W., Kuchta, M. & Martin, R. (2013). *Open Pit Mine Planning
       and Design*, 3rd ed. CRC Press, Ch. 7.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Match Factor
# ---------------------------------------------------------------------------


def match_factor(
    n_trucks: int,
    truck_cycle: float,
    n_loaders: int,
    loader_cycle: float,
) -> dict:
    """Compute the truck-loader Match Factor (MF).

    The Match Factor quantifies the balance between the truck fleet and the
    loading unit(s):

    .. math::

        MF = \\frac{N_t \\times C_l}{N_l \\times C_t}

    - MF = 1.0 : Perfect match (no idle time).
    - MF < 1.0 : Truck-limited (loader waits).
    - MF > 1.0 : Loader-limited (trucks queue).

    Parameters
    ----------
    n_trucks : int
        Number of trucks in the fleet.  Must be >= 1.
    truck_cycle : float
        Average truck cycle time in minutes.  Must be positive.
    n_loaders : int
        Number of loaders (or shovels).  Must be >= 1.
    loader_cycle : float
        Average loader cycle time per truck (time to fill one truck), in
        minutes.  Must be positive.

    Returns
    -------
    dict
        Dictionary with:

        - ``"mf"`` : float -- Match Factor value.
        - ``"status"`` : str -- ``"balanced"``, ``"truck_limited"``, or
          ``"loader_limited"``.
        - ``"bottleneck"`` : str -- ``"none"``, ``"trucks"``, or
          ``"loaders"``.

    Examples
    --------
    >>> result = match_factor(5, 30, 1, 5)
    >>> round(result["mf"], 3)
    0.833
    >>> result["status"]
    'truck_limited'

    References
    ----------
    .. [1] SME Mining Engineering Handbook, 3rd ed. (2011), Ch. 9.5.
    """
    if n_trucks < 1:
        raise ValueError(f"'n_trucks' must be >= 1, got {n_trucks}.")
    if n_loaders < 1:
        raise ValueError(f"'n_loaders' must be >= 1, got {n_loaders}.")
    validate_positive(truck_cycle, "truck_cycle")
    validate_positive(loader_cycle, "loader_cycle")

    mf = (n_trucks * loader_cycle) / (n_loaders * truck_cycle)

    if abs(mf - 1.0) < 1e-9:
        status = "balanced"
        bottleneck = "none"
    elif mf < 1.0:
        status = "truck_limited"
        bottleneck = "trucks"
    else:
        status = "loader_limited"
        bottleneck = "loaders"

    return {
        "mf": mf,
        "status": status,
        "bottleneck": bottleneck,
    }


# ---------------------------------------------------------------------------
# Optimal Fleet Sizing
# ---------------------------------------------------------------------------


def optimal_fleet(
    truck_cycle: float,
    loader_cycle: float,
    target_production: float,
    truck_capacity: float,
    availability: float = 0.85,
    utilization: float = 0.90,
) -> dict:
    """Determine the optimal number of trucks to match one loader and meet
    a target production rate.

    The base fleet size for MF = 1.0 is:

    .. math::

        N_t^{base} = \\lceil C_t / C_l \\rceil

    Production is then estimated and, if insufficient, the fleet is scaled
    upward:

    .. math::

        P = N_t \\times Q \\times \\frac{60}{C_t} \\times A \\times U

    Parameters
    ----------
    truck_cycle : float
        Average truck cycle time in minutes.  Must be positive.
    loader_cycle : float
        Average loader cycle time per truck (fill time), in minutes.
        Must be positive.
    target_production : float
        Required production rate in t/h.  Must be positive.
    truck_capacity : float
        Truck payload capacity in tonnes.  Must be positive.
    availability : float, optional
        Mechanical availability as a fraction in (0, 1] (default 0.85).
    utilization : float, optional
        Operating utilization as a fraction in (0, 1] (default 0.90).

    Returns
    -------
    dict
        Dictionary with:

        - ``"n_trucks"`` : int -- Required number of trucks.
        - ``"production"`` : float -- Estimated production in t/h.
        - ``"match_factor"`` : float -- Resulting match factor (with 1
          loader).

    Examples
    --------
    >>> result = optimal_fleet(30.0, 5.0, 500.0, 150.0)
    >>> result["n_trucks"]
    6
    >>> round(result["production"], 1)
    1377.0

    References
    ----------
    .. [1] SME Mining Engineering Handbook, 3rd ed. (2011), Ch. 9.5.
    """
    validate_positive(truck_cycle, "truck_cycle")
    validate_positive(loader_cycle, "loader_cycle")
    validate_positive(target_production, "target_production")
    validate_positive(truck_capacity, "truck_capacity")
    validate_range(availability, 0.0, 1.0, "availability")
    validate_range(utilization, 0.0, 1.0, "utilization")
    if availability == 0:
        raise ValueError("'availability' must be > 0.")
    if utilization == 0:
        raise ValueError("'utilization' must be > 0.")

    # Base fleet for MF = 1
    n_trucks_base = math.ceil(truck_cycle / loader_cycle)

    # Estimate production with base fleet
    n_trucks = n_trucks_base
    production = n_trucks * truck_capacity * (60.0 / truck_cycle) * availability * utilization

    # If base fleet is insufficient, increase until target is met
    while production < target_production:
        n_trucks += 1
        production = n_trucks * truck_capacity * (60.0 / truck_cycle) * availability * utilization

    mf = (n_trucks * loader_cycle) / truck_cycle

    return {
        "n_trucks": n_trucks,
        "production": production,
        "match_factor": mf,
    }
