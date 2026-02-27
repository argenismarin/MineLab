"""Thickening and sedimentation calculations.

Kynch batch settling analysis, Talmage-Fitch thickener sizing,
Coe-Clevenger unit area, and flocculant dosage.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive


def kynch_analysis(
    times: np.ndarray,
    heights: np.ndarray,
) -> dict:
    """Kynch batch settling test analysis.

    Parameters
    ----------
    times : np.ndarray
        Settling times (min).
    heights : np.ndarray
        Interface heights (m or cm — consistent units).

    Returns
    -------
    dict
        Keys: ``"settling_rates"`` (array of rates),
        ``"concentrations"`` (array of solids concentrations as ratio
        to initial), ``"critical_time"`` (min).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0, 5, 10, 20, 30, 60])
    >>> h = np.array([1.0, 0.8, 0.65, 0.45, 0.35, 0.3])
    >>> result = kynch_analysis(t, h)
    >>> len(result["settling_rates"]) > 0
    True

    References
    ----------
    .. [1] Kynch, G.J. (1952). "A theory of sedimentation." Trans.
       Faraday Soc., 48, 166-176.
    """
    times = np.asarray(times, dtype=float)
    heights = np.asarray(heights, dtype=float)

    # Settling rate = -dh/dt
    rates = -np.diff(heights) / np.diff(times)

    # Concentration ratio: C/C0 = H0/H
    h0 = heights[0]
    concentrations = h0 / heights[1:]  # skip t=0

    # Critical time: where settling rate changes significantly
    if len(rates) > 1:
        rate_changes = np.abs(np.diff(rates))
        critical_idx = np.argmax(rate_changes) + 1
        critical_time = float(times[critical_idx])
    else:
        critical_time = float(times[-1])

    return {
        "settling_rates": rates,
        "concentrations": concentrations,
        "critical_time": critical_time,
    }


def talmage_fitch(
    initial_height: float,
    underflow_conc: float,
    initial_conc: float,
    settling_rate: float,
    feed_rate: float,
) -> dict:
    """Talmage-Fitch thickener area calculation.

    Parameters
    ----------
    initial_height : float
        Initial settling test height (m).
    underflow_conc : float
        Target underflow solids concentration (fraction, 0-1).
    initial_conc : float
        Initial solids concentration (fraction, 0-1).
    settling_rate : float
        Initial settling rate (m/h).
    feed_rate : float
        Volumetric feed rate (m^3/h).

    Returns
    -------
    dict
        Keys: ``"unit_area"`` (m^2 per t/h solids),
        ``"thickener_area"`` (m^2), ``"diameter"`` (m).

    Examples
    --------
    >>> result = talmage_fitch(1.0, 0.5, 0.1, 0.5, 100)
    >>> result["thickener_area"] > 0
    True

    References
    ----------
    .. [1] Talmage, W.P. & Fitch, E.B. (1955). "Determining thickener
       unit areas." Ind. Eng. Chem., 47(1), 38-41.
    """
    validate_positive(initial_height, "initial_height")
    validate_positive(underflow_conc, "underflow_conc")
    validate_positive(initial_conc, "initial_conc")
    validate_positive(settling_rate, "settling_rate")
    validate_positive(feed_rate, "feed_rate")

    # Unit area = H0 / (settling_rate * Cu)
    # where Cu = underflow concentration
    tu = initial_height * (1 / initial_conc - 1 / underflow_conc) / settling_rate

    # Area = F * tu / H0
    area = feed_rate * tu / initial_height

    diameter = np.sqrt(4 * area / np.pi)

    solids_rate = feed_rate * initial_conc
    unit_area = area / solids_rate if solids_rate > 0 else 0

    return {
        "unit_area": float(unit_area),
        "thickener_area": float(area),
        "diameter": float(diameter),
    }


def coe_clevenger(
    settling_rates: np.ndarray,
    concentrations: np.ndarray,
    underflow_conc: float,
    feed_rate: float,
    initial_conc: float,
) -> dict:
    """Coe-Clevenger unit area method for thickener design.

    Parameters
    ----------
    settling_rates : np.ndarray
        Settling rates at various concentrations (m/h).
    concentrations : np.ndarray
        Corresponding solids concentrations (fraction, 0-1).
    underflow_conc : float
        Target underflow concentration (fraction, 0-1).
    feed_rate : float
        Volumetric feed rate (m^3/h).
    initial_conc : float
        Initial solids concentration (fraction, 0-1).

    Returns
    -------
    dict
        Keys: ``"unit_area"`` (m^2/(t/h)), ``"thickener_area"`` (m^2),
        ``"controlling_concentration"`` (fraction).

    Examples
    --------
    >>> import numpy as np
    >>> rates = np.array([0.5, 0.3, 0.1, 0.05])
    >>> concs = np.array([0.05, 0.1, 0.2, 0.3])
    >>> result = coe_clevenger(rates, concs, 0.4, 100, 0.05)
    >>> result["thickener_area"] > 0
    True

    References
    ----------
    .. [1] Coe, H.S. & Clevenger, G.H. (1916). Methods for determining
       the capacities of slime-settling tanks.
    """
    settling_rates = np.asarray(settling_rates, dtype=float)
    concentrations = np.asarray(concentrations, dtype=float)

    # Unit area for each concentration:
    # UA = (1/C - 1/Cu) / v
    unit_areas = (1 / concentrations - 1 / underflow_conc) / settling_rates

    # Controlling concentration = max unit area
    idx = np.argmax(unit_areas)
    max_ua = unit_areas[idx]
    controlling_conc = concentrations[idx]

    solids_rate = feed_rate * initial_conc
    area = max_ua * solids_rate

    return {
        "unit_area": float(max_ua),
        "thickener_area": float(area),
        "controlling_concentration": float(controlling_conc),
    }


def flocculant_dosage(
    feed_rate: float,
    solids_conc: float,
    dose_gpt: float,
) -> float:
    """Flocculant consumption calculation.

    Parameters
    ----------
    feed_rate : float
        Feed slurry rate (m^3/h).
    solids_conc : float
        Solids concentration in feed (fraction, 0-1).
    dose_gpt : float
        Flocculant dose (g/t of solids).

    Returns
    -------
    float
        Flocculant consumption rate (kg/h).

    Examples
    --------
    >>> round(flocculant_dosage(100, 0.1, 20), 1)
    0.2

    References
    ----------
    .. [1] Industrial practice.
    """
    validate_positive(feed_rate, "feed_rate")
    validate_positive(dose_gpt, "dose_gpt")

    # Solids rate (t/h, assuming ~1 t/m^3 slurry density for simplicity)
    solids_rate = feed_rate * solids_conc  # approximate t/h

    # Consumption = solids_rate * dose / 1000 (g → kg)
    consumption = solids_rate * dose_gpt / 1000

    return float(consumption)
