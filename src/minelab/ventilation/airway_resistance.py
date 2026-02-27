"""Airway resistance calculations for mine ventilation networks.

This module provides functions to compute friction-based resistance in mine
airways, pressure drops, series/parallel equivalent resistances, and the
natural ventilation pressure arising from temperature differentials between
surface and underground.

References
----------
.. [1] McPherson, M.J. (1993). *Subsurface Ventilation and Environmental
       Engineering*, 1st ed. Chapman & Hall, Chapters 5, 7, 8.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import validate_non_negative, validate_positive

# ---------------------------------------------------------------------------
# Atkinson resistance
# ---------------------------------------------------------------------------


def atkinson_resistance(
    k: float,
    length: float,
    perimeter: float,
    area: float,
) -> float:
    """Compute airway resistance using the Atkinson equation.

    .. math::

        R = \\frac{k \\cdot L \\cdot \\mathrm{Per}}{A^3}

    Parameters
    ----------
    k : float
        Atkinson friction factor (kg/m^3).
    length : float
        Airway length (m).
    perimeter : float
        Airway perimeter (m).
    area : float
        Airway cross-sectional area (m^2).

    Returns
    -------
    float
        Airway resistance in Ns^2/m^8.

    Raises
    ------
    ValueError
        If any parameter is not positive.

    Examples
    --------
    >>> round(atkinson_resistance(0.012, 500, 12, 9), 4)
    0.0988

    References
    ----------
    .. [1] McPherson (1993), Ch. 5, Eq. 5.4.
    """
    validate_positive(k, "k")
    validate_positive(length, "length")
    validate_positive(perimeter, "perimeter")
    validate_positive(area, "area")
    return k * length * perimeter / (area**3)


# ---------------------------------------------------------------------------
# Pressure drop
# ---------------------------------------------------------------------------


def pressure_drop(resistance: float, airflow: float) -> float:
    """Compute the pressure drop across an airway.

    .. math::

        \\Delta P = R \\cdot Q^2

    Parameters
    ----------
    resistance : float
        Airway resistance R (Ns^2/m^8).  Must be non-negative.
    airflow : float
        Volume airflow rate Q (m^3/s).  Must be non-negative.

    Returns
    -------
    float
        Pressure drop in Pa.

    Raises
    ------
    ValueError
        If *resistance* or *airflow* is negative.

    Examples
    --------
    >>> pressure_drop(0.5, 50)
    1250.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 5, Eq. 5.1.
    """
    validate_non_negative(resistance, "resistance")
    validate_non_negative(airflow, "airflow")
    return resistance * airflow**2


# ---------------------------------------------------------------------------
# Friction factor from roughness
# ---------------------------------------------------------------------------


def friction_factor_from_roughness(
    roughness: float,
    hydraulic_diam: float,
) -> float:
    """Estimate the Atkinson friction factor from surface roughness height.

    Uses the simplified empirical relationship for mine airways:

    .. math::

        k \\approx 0.6 \\cdot \\frac{e}{D_h}

    where *e* is the roughness height and *D_h* is the hydraulic diameter.

    Parameters
    ----------
    roughness : float
        Roughness height *e* (m).  Must be positive.
    hydraulic_diam : float
        Hydraulic diameter of the airway (m).  Must be positive.

    Returns
    -------
    float
        Estimated Atkinson friction factor (kg/m^3).

    Raises
    ------
    ValueError
        If any parameter is not positive.

    Examples
    --------
    >>> round(friction_factor_from_roughness(0.05, 3.0), 4)
    0.01

    References
    ----------
    .. [1] McPherson (1993), Ch. 5, Table 5.1 and discussion.
    """
    validate_positive(roughness, "roughness")
    validate_positive(hydraulic_diam, "hydraulic_diam")
    return 0.6 * roughness / hydraulic_diam


# ---------------------------------------------------------------------------
# Series resistance
# ---------------------------------------------------------------------------


def series_resistance(resistances: list[float]) -> float:
    """Compute the total resistance of airways in series.

    .. math::

        R_{\\text{total}} = \\sum_i R_i

    Parameters
    ----------
    resistances : list of float
        Individual airway resistances (Ns^2/m^8).  All must be non-negative.

    Returns
    -------
    float
        Total equivalent resistance in Ns^2/m^8.

    Raises
    ------
    ValueError
        If the list is empty or any resistance is negative.

    Examples
    --------
    >>> series_resistance([1.0, 2.0, 3.0])
    6.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 7, Sec. 7.3.1.
    """
    if not resistances:
        raise ValueError("'resistances' must contain at least one element.")
    for i, r in enumerate(resistances):
        validate_non_negative(r, f"resistances[{i}]")
    return float(sum(resistances))


# ---------------------------------------------------------------------------
# Parallel resistance
# ---------------------------------------------------------------------------


def parallel_resistance(resistances: list[float]) -> float:
    """Compute the total resistance of airways in parallel.

    For parallel airways the relationship is:

    .. math::

        \\frac{1}{\\sqrt{R_{\\text{total}}}} = \\sum_i \\frac{1}{\\sqrt{R_i}}

    Parameters
    ----------
    resistances : list of float
        Individual airway resistances (Ns^2/m^8).  All must be positive.

    Returns
    -------
    float
        Total equivalent resistance in Ns^2/m^8.

    Raises
    ------
    ValueError
        If the list is empty or any resistance is not positive.

    Examples
    --------
    >>> parallel_resistance([4.0, 4.0])
    1.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 7, Sec. 7.3.2.
    """
    if not resistances:
        raise ValueError("'resistances' must contain at least one element.")
    for i, r in enumerate(resistances):
        validate_positive(r, f"resistances[{i}]")
    inv_sqrt_sum = sum(1.0 / math.sqrt(r) for r in resistances)
    return 1.0 / (inv_sqrt_sum**2)


# ---------------------------------------------------------------------------
# Natural ventilation pressure
# ---------------------------------------------------------------------------


def natural_ventilation_pressure(
    depths: list[float],
    temps_surface: list[float],
    temps_underground: list[float],
) -> float:
    """Estimate the natural ventilation pressure from temperature differentials.

    The NVP arises because air columns at different temperatures have
    different densities.  A simplified approximation is used:

    .. math::

        \\text{NVP} \\approx 0.0034 \\cdot H \\cdot \\Delta T

    where *H* is the depth (m) and *DeltaT* = *T_underground* - *T_surface*
    (in degrees Celsius or Kelvin).

    When multiple depth/temperature pairs are provided, the total NVP is
    the sum of individual contributions.

    Parameters
    ----------
    depths : list of float
        Depths for each segment (m).  All must be positive.
    temps_surface : list of float
        Surface (intake) air temperatures for each segment (deg C).
    temps_underground : list of float
        Underground (return) air temperatures for each segment (deg C).

    Returns
    -------
    float
        Natural ventilation pressure in Pa.  Positive values indicate that
        the thermal draft assists ventilation (underground warmer than
        surface).

    Raises
    ------
    ValueError
        If lists have mismatched lengths, are empty, or depths are not
        positive.

    Examples
    --------
    >>> round(natural_ventilation_pressure([500], [15], [30]), 2)
    25.5

    References
    ----------
    .. [1] McPherson (1993), Ch. 8, Sec. 8.3.
    """
    if not depths:
        raise ValueError("'depths' must contain at least one element.")
    if len(depths) != len(temps_surface) or len(depths) != len(temps_underground):
        raise ValueError(
            "All input lists must have the same length. "
            f"Got depths={len(depths)}, temps_surface={len(temps_surface)}, "
            f"temps_underground={len(temps_underground)}."
        )
    for i, d in enumerate(depths):
        validate_positive(d, f"depths[{i}]")

    nvp_total = 0.0
    for h, t_s, t_u in zip(depths, temps_surface, temps_underground, strict=True):
        delta_t = t_u - t_s
        nvp_total += 0.0034 * h * delta_t

    return nvp_total
