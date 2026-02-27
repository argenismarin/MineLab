"""Equipment productivity calculations for surface and underground mining.

This module provides functions for estimating fleet and individual equipment
productivity rates, and Overall Equipment Effectiveness (OEE).

References
----------
.. [1] SME Mining Engineering Handbook, 3rd ed. (2011). Society for Mining,
       Metallurgy & Exploration, Ch. 9.
.. [2] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Fleet Productivity
# ---------------------------------------------------------------------------


def fleet_productivity(
    n_trucks: int,
    truck_capacity: float,
    cycle_time: float,
    availability: float = 0.85,
    utilization: float = 0.90,
) -> float:
    """Estimate hourly truck fleet production.

    .. math::

        P = N_t \\times Q \\times \\frac{60}{C_t} \\times A \\times U

    Parameters
    ----------
    n_trucks : int
        Number of trucks in the fleet.  Must be >= 1.
    truck_capacity : float
        Truck payload capacity in tonnes.  Must be positive.
    cycle_time : float
        Average truck cycle time in minutes.  Must be positive.
    availability : float, optional
        Mechanical availability as a fraction in (0, 1] (default 0.85).
    utilization : float, optional
        Operating utilization as a fraction in (0, 1] (default 0.90).

    Returns
    -------
    float
        Fleet production rate in t/h.

    Examples
    --------
    >>> round(fleet_productivity(5, 150, 30, 0.85, 0.90), 1)
    1147.5

    References
    ----------
    .. [1] SME Mining Engineering Handbook, 3rd ed. (2011), Ch. 9.5.
    """
    if n_trucks < 1:
        raise ValueError(f"'n_trucks' must be >= 1, got {n_trucks}.")
    validate_positive(truck_capacity, "truck_capacity")
    validate_positive(cycle_time, "cycle_time")
    validate_range(availability, 0.0, 1.0, "availability")
    validate_range(utilization, 0.0, 1.0, "utilization")

    return n_trucks * truck_capacity * (60.0 / cycle_time) * availability * utilization


# ---------------------------------------------------------------------------
# Excavator Productivity
# ---------------------------------------------------------------------------


def excavator_productivity(
    bucket_size: float,
    fill_factor: float,
    cycle_time: float,
    material_density: float,
    availability: float = 0.85,
) -> float:
    """Estimate hourly excavator (shovel/loader) production.

    .. math::

        P = \\frac{3600 \\times V_b \\times f_f \\times \\rho}{C_t} \\times A

    Parameters
    ----------
    bucket_size : float
        Bucket capacity in m^3 (heaped or rated).  Must be positive.
    fill_factor : float
        Bucket fill factor as a fraction in (0, 1] (typically 0.80--0.95).
    cycle_time : float
        Excavator cycle time (swing-load-swing-dump) in seconds.
        Must be positive.
    material_density : float
        In-situ or loose material density in t/m^3.  Must be positive.
    availability : float, optional
        Mechanical availability as a fraction in (0, 1] (default 0.85).

    Returns
    -------
    float
        Excavator production rate in t/h.

    Examples
    --------
    >>> round(excavator_productivity(15.0, 0.85, 30.0, 2.5, 0.90), 1)
    3442.5

    References
    ----------
    .. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
    """
    validate_positive(bucket_size, "bucket_size")
    validate_range(fill_factor, 0.0, 1.0, "fill_factor")
    if fill_factor == 0:
        raise ValueError("'fill_factor' must be > 0.")
    validate_positive(cycle_time, "cycle_time")
    validate_positive(material_density, "material_density")
    validate_range(availability, 0.0, 1.0, "availability")

    return (3600.0 * bucket_size * fill_factor * material_density / cycle_time) * availability


# ---------------------------------------------------------------------------
# Overall Equipment Effectiveness
# ---------------------------------------------------------------------------


def oee(
    availability: float,
    utilization: float,
    efficiency: float,
) -> float:
    """Compute Overall Equipment Effectiveness (OEE).

    .. math::

        OEE = A \\times U \\times E

    OEE is a dimensionless metric widely used in industry to benchmark
    equipment performance.  World-class OEE is typically >= 0.85.

    Parameters
    ----------
    availability : float
        Mechanical availability as a fraction in [0, 1].
    utilization : float
        Operating utilization as a fraction in [0, 1].
    efficiency : float
        Performance efficiency (actual output / ideal output) as a
        fraction in [0, 1].

    Returns
    -------
    float
        OEE as a fraction in [0, 1].

    Examples
    --------
    >>> round(oee(0.90, 0.85, 0.95), 4)
    0.7268

    References
    ----------
    .. [1] Nakajima, S. (1988). *Introduction to TPM*. Productivity Press.
    """
    validate_range(availability, 0.0, 1.0, "availability")
    validate_range(utilization, 0.0, 1.0, "utilization")
    validate_range(efficiency, 0.0, 1.0, "efficiency")

    return availability * utilization * efficiency
