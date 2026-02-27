"""Fuel consumption and cost estimation for mining equipment.

This module provides functions for estimating diesel fuel consumption rates
and the fuel cost component per tonne of material moved.

References
----------
.. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
.. [2] Hustrulid, W., Kuchta, M. & Martin, R. (2013). *Open Pit Mine Planning
       and Design*, 3rd ed. CRC Press, Ch. 7.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Fuel Consumption Rate
# ---------------------------------------------------------------------------


def fuel_consumption_rate(
    engine_kw: float,
    load_factor: float,
    specific_consumption: float = 0.24,
) -> float:
    """Estimate diesel fuel consumption rate.

    .. math::

        Q_f = P_{engine} \\times LF \\times SFC

    Parameters
    ----------
    engine_kw : float
        Rated engine power in kW.  Must be positive.
    load_factor : float
        Engine load factor as a fraction in (0, 1] (typically 0.40--0.70
        for haul trucks).
    specific_consumption : float, optional
        Specific fuel consumption in L/kWh (default 0.24 for typical
        diesel engines).  Must be positive.

    Returns
    -------
    float
        Fuel consumption rate in L/h.

    Examples
    --------
    >>> round(fuel_consumption_rate(1500, 0.55), 1)
    198.0

    References
    ----------
    .. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.,
           Ch. on fuel consumption estimation.
    """
    validate_positive(engine_kw, "engine_kw")
    validate_range(load_factor, 0.0, 1.0, "load_factor")
    if load_factor == 0:
        raise ValueError("'load_factor' must be > 0.")
    validate_positive(specific_consumption, "specific_consumption")

    return engine_kw * load_factor * specific_consumption


# ---------------------------------------------------------------------------
# Fuel Cost per Tonne
# ---------------------------------------------------------------------------


def fuel_cost_per_tonne(
    consumption_rate: float,
    fuel_price: float,
    productivity: float,
) -> float:
    """Estimate the fuel cost component per tonne of material moved.

    .. math::

        C_f = \\frac{Q_f \\times p_f}{P}

    Parameters
    ----------
    consumption_rate : float
        Fuel consumption rate in L/h.  Must be positive.
    fuel_price : float
        Fuel price in USD/L (or other currency per litre).  Must be
        positive.
    productivity : float
        Equipment or fleet productivity in t/h.  Must be positive.

    Returns
    -------
    float
        Fuel cost in USD/t (or corresponding currency per tonne).

    Examples
    --------
    >>> round(fuel_cost_per_tonne(198.0, 1.20, 382.5), 3)
    0.621

    References
    ----------
    .. [1] Industrial practice; see also Hustrulid et al. (2013), Ch. 7.
    """
    validate_positive(consumption_rate, "consumption_rate")
    validate_positive(fuel_price, "fuel_price")
    validate_positive(productivity, "productivity")

    return (consumption_rate * fuel_price) / productivity
