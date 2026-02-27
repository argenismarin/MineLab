"""Dust emission estimation and atmospheric dispersion.

This module provides the USEPA AP-42 unpaved-road emission factor for haul
roads and the Gaussian plume dispersion model for predicting downwind
pollutant concentrations.

References
----------
.. [1] USEPA (2006). AP-42, *Compilation of Air Pollutant Emission Factors*,
       Volume I, Chapter 13.2.2 -- Unpaved Roads.
.. [2] Turner, D.B. (1970). *Workbook of Atmospheric Dispersion Estimates*.
       USEPA Office of Air Programs, Publication No. AP-26.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# AP-42 Haul Road Emission Factor
# ---------------------------------------------------------------------------

# AP-42 constants for PM10 on unpaved roads
_K_PM10 = 1.5  # empirical multiplier (kg/VKT)
_A_SILT = 0.9  # exponent for silt fraction
_B_WEIGHT = 0.45  # exponent for vehicle weight


def emission_factor_haul_roads(
    silt_pct: float,
    vehicle_weight: float,
    speed: float = 30.0,
    moisture: float = 5.0,
) -> float:
    """Compute PM10 emission factor for unpaved haul roads (AP-42).

    .. math::

        EF = k \\left(\\frac{s}{12}\\right)^{a}
                \\left(\\frac{W}{3}\\right)^{b}

    where *k* = 1.5 kg/VKT (PM10), *a* = 0.9, *b* = 0.45, *s* is the
    road-surface silt content (%), and *W* is the mean vehicle weight
    (tonnes).

    Parameters
    ----------
    silt_pct : float
        Road-surface silt content in percent (typically 1--25).
        Must be positive.
    vehicle_weight : float
        Mean vehicle weight in tonnes.  Must be positive.
    speed : float, optional
        Average vehicle speed in km/h (used for reference context only
        in this simplified form).  Default is 30.0.
    moisture : float, optional
        Road-surface moisture content in percent.  Default is 5.0.
        Must be non-negative.

    Returns
    -------
    float
        Emission factor in kg per vehicle-kilometre travelled (kg/VKT).

    Examples
    --------
    >>> round(emission_factor_haul_roads(12, 3), 2)
    1.5

    >>> ef = emission_factor_haul_roads(8, 100)
    >>> ef > 0
    True

    References
    ----------
    .. [1] USEPA AP-42, Ch. 13.2.2 (2006).
    """
    validate_positive(silt_pct, "silt_pct")
    validate_positive(vehicle_weight, "vehicle_weight")
    validate_positive(speed, "speed")
    validate_range(moisture, 0, 100, "moisture")

    ef = _K_PM10 * (silt_pct / 12.0) ** _A_SILT * (vehicle_weight / 3.0) ** _B_WEIGHT
    return ef


# ---------------------------------------------------------------------------
# Gaussian Plume Dispersion
# ---------------------------------------------------------------------------


def gaussian_plume(
    Q: float,  # noqa: N803
    u: float,
    sigma_y: float,
    sigma_z: float,
    H: float,  # noqa: N803
    x: float,
    y: float,
    z: float = 0.0,
) -> float:
    """Compute ground-level concentration using the Gaussian plume model.

    .. math::

        C = \\frac{Q}{2\\pi\\, u\\, \\sigma_y\\, \\sigma_z}
            \\exp\\!\\left(-\\frac{y^2}{2\\sigma_y^2}\\right)
            \\left[
                \\exp\\!\\left(-\\frac{(z - H)^2}{2\\sigma_z^2}\\right)
              + \\exp\\!\\left(-\\frac{(z + H)^2}{2\\sigma_z^2}\\right)
            \\right]

    The dispersion parameters *sigma_y* and *sigma_z* are typically
    functions of downwind distance *x* and atmospheric stability class,
    but are passed directly here for flexibility.

    Parameters
    ----------
    Q : float
        Source emission rate (g/s or ug/s -- units carry through to the
        output concentration).  Must be positive.
    u : float
        Wind speed at effective stack height in m/s.  Must be positive.
    sigma_y : float
        Horizontal dispersion parameter in metres.  Must be positive.
    sigma_z : float
        Vertical dispersion parameter in metres.  Must be positive.
    H : float
        Effective source (stack) height in metres.  Must be non-negative.
    x : float
        Downwind distance in metres.  Must be positive.  (Not used
        directly in the equation but retained for API consistency --
        *sigma_y* and *sigma_z* are functions of *x*.)
    y : float
        Crosswind offset in metres (can be negative).
    z : float, optional
        Receptor height above ground in metres.  Default is 0.0
        (ground-level).

    Returns
    -------
    float
        Pollutant concentration at the receptor in the same mass-per-volume
        unit implied by *Q* (e.g. g/m3 if *Q* is in g/s).

    Examples
    --------
    >>> # Centreline ground-level concentration
    >>> c = gaussian_plume(Q=10, u=5, sigma_y=50, sigma_z=30,
    ...                    H=20, x=500, y=0, z=0)
    >>> c > 0
    True

    References
    ----------
    .. [1] Turner, D.B. (1970). *Workbook of Atmospheric Dispersion
           Estimates*. AP-26.
    .. [2] USEPA AP-42 (2006), background dispersion methodology.
    """
    validate_positive(Q, "Q")
    validate_positive(u, "u")
    validate_positive(sigma_y, "sigma_y")
    validate_positive(sigma_z, "sigma_z")
    validate_range(H, 0, float("inf"), "H")
    validate_positive(x, "x")
    validate_range(z, 0, float("inf"), "z")

    coeff = Q / (2.0 * math.pi * u * sigma_y * sigma_z)
    exp_y = math.exp(-(y**2) / (2.0 * sigma_y**2))
    exp_z1 = math.exp(-((z - H) ** 2) / (2.0 * sigma_z**2))
    exp_z2 = math.exp(-((z + H) ** 2) / (2.0 * sigma_z**2))

    return coeff * exp_y * (exp_z1 + exp_z2)
