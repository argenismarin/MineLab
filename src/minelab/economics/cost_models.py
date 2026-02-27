"""Mining cost estimation models.

This module provides capital and operating cost estimation functions commonly
used in prefeasibility and feasibility studies for mining projects.

References
----------
.. [1] Mular, A.L. & Poulin, R. (1998). *Capcosts: A Handbook for Estimating
       Mining and Mineral Processing Equipment Costs and Capital
       Expenditures*. CIM.
.. [2] Taylor, H.K. (1977). Mine valuation and feasibility studies.
       *Mineral Industry Costs*, pp. 1--17. Northwest Mining Assoc.
.. [3] Hustrulid, W. et al. (2013). *Open Pit Mine Planning and Design*,
       3rd ed. CRC Press.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Capital Cost — Six-Tenths Rule
# ---------------------------------------------------------------------------


def capex_estimate(
    capacity: float,
    base_cost: float,
    base_capacity: float,
    exponent: float = 0.6,
) -> float:
    """Estimate capital cost using the six-tenths (power-sizing) rule.

    cost = base_cost * (capacity / base_capacity) ^ exponent

    Parameters
    ----------
    capacity : float
        Target plant/equipment capacity.  Must be > 0.
    base_cost : float
        Known cost at *base_capacity*.  Must be > 0.
    base_capacity : float
        Reference capacity corresponding to *base_cost*.  Must be > 0.
    exponent : float, optional
        Scaling exponent (default 0.6, the classic Williams rule).

    Returns
    -------
    float
        Estimated capital cost.

    Raises
    ------
    ValueError
        If any capacity or cost is non-positive.

    Examples
    --------
    >>> round(capex_estimate(5000, 10_000_000, 2000), 2)
    17328621.08

    References
    ----------
    .. [1] Mular & Poulin (1998), Sec. 2.
    """
    if capacity <= 0 or base_cost <= 0 or base_capacity <= 0:
        raise ValueError("Capacity and cost values must be positive.")
    return float(base_cost * (capacity / base_capacity) ** exponent)


# ---------------------------------------------------------------------------
# Operating Cost Per Tonne
# ---------------------------------------------------------------------------


def opex_per_tonne(
    mining: float,
    processing: float,
    ga: float,
    other: float = 0.0,
) -> float:
    """Compute total operating expenditure per tonne.

    Parameters
    ----------
    mining : float
        Mining cost per tonne (USD/t).  Must be >= 0.
    processing : float
        Processing cost per tonne (USD/t).  Must be >= 0.
    ga : float
        General & administrative cost per tonne (USD/t).  Must be >= 0.
    other : float, optional
        Any additional cost per tonne (default 0).

    Returns
    -------
    float
        Total OPEX per tonne.

    Raises
    ------
    ValueError
        If any cost component is negative.

    Examples
    --------
    >>> opex_per_tonne(2.50, 8.00, 1.50)
    12.0
    """
    if mining < 0 or processing < 0 or ga < 0 or other < 0:
        raise ValueError("Cost components must be non-negative.")
    return float(mining + processing + ga + other)


# ---------------------------------------------------------------------------
# Stripping Cost
# ---------------------------------------------------------------------------


def stripping_cost(
    waste_tonnes: float,
    ore_tonnes: float,
    cost_per_tonne: float,
) -> float:
    """Compute the waste stripping cost allocated to each tonne of ore.

    Parameters
    ----------
    waste_tonnes : float
        Total tonnes of waste mined.  Must be >= 0.
    ore_tonnes : float
        Total tonnes of ore mined.  Must be > 0.
    cost_per_tonne : float
        Cost per tonne of waste removal (USD/t).  Must be >= 0.

    Returns
    -------
    float
        Stripping cost allocated per tonne of ore (USD/t).

    Examples
    --------
    >>> stripping_cost(3_000_000, 1_000_000, 2.50)
    7.5
    """
    if waste_tonnes < 0 or cost_per_tonne < 0:
        raise ValueError("Waste tonnes and cost must be non-negative.")
    if ore_tonnes <= 0:
        raise ValueError("Ore tonnes must be positive.")
    return float(waste_tonnes * cost_per_tonne / ore_tonnes)


# ---------------------------------------------------------------------------
# Straight-Line Depreciation
# ---------------------------------------------------------------------------


def depreciation_straight_line(
    capex: float,
    salvage: float,
    life: int,
) -> float:
    """Annual depreciation using the straight-line method.

    Parameters
    ----------
    capex : float
        Initial capital expenditure.  Must be > 0.
    salvage : float
        Salvage (residual) value at end of life.  Must be >= 0.
    life : int
        Useful life in years.  Must be >= 1.

    Returns
    -------
    float
        Annual depreciation amount.

    Raises
    ------
    ValueError
        If inputs are out of valid range, or salvage > capex.

    Examples
    --------
    >>> depreciation_straight_line(1_000_000, 100_000, 10)
    90000.0
    """
    if capex <= 0:
        raise ValueError("CAPEX must be positive.")
    if salvage < 0:
        raise ValueError("Salvage value must be non-negative.")
    if salvage > capex:
        raise ValueError("Salvage value cannot exceed CAPEX.")
    if life < 1:
        raise ValueError("Useful life must be at least 1 year.")
    return float((capex - salvage) / life)


# ---------------------------------------------------------------------------
# Declining Balance Depreciation
# ---------------------------------------------------------------------------


def depreciation_declining_balance(
    capex: float,
    rate: float,
    years: int,
) -> list[float]:
    """Compute depreciation schedule using the declining-balance method.

    Parameters
    ----------
    capex : float
        Initial capital expenditure.  Must be > 0.
    rate : float
        Annual depreciation rate as a decimal (e.g. 0.20 for 20 %).
        Must be in (0, 1].
    years : int
        Number of years to compute.  Must be >= 1.

    Returns
    -------
    list of float
        Annual depreciation amounts for each year.

    Raises
    ------
    ValueError
        If inputs are out of valid range.

    Examples
    --------
    >>> depreciation_declining_balance(1_000_000, 0.20, 3)
    [200000.0, 160000.0, 128000.0]

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 7.3.
    """
    if capex <= 0:
        raise ValueError("CAPEX must be positive.")
    if not 0 < rate <= 1:
        raise ValueError("Depreciation rate must be in (0, 1].")
    if years < 1:
        raise ValueError("Number of years must be at least 1.")

    schedule: list[float] = []
    book_value = float(capex)
    for _ in range(years):
        dep = book_value * rate
        schedule.append(dep)
        book_value -= dep
    return schedule


# ---------------------------------------------------------------------------
# Taylor's Rule
# ---------------------------------------------------------------------------


def taylor_rule(reserves_mt: float) -> float:
    """Estimate annual production capacity using Taylor's Rule (1977).

    capacity (Mt/yr) = 0.25 * reserves_mt ^ 0.75

    Parameters
    ----------
    reserves_mt : float
        Total ore reserves in million tonnes (Mt).  Must be > 0.

    Returns
    -------
    float
        Estimated annual capacity in million tonnes per year (Mt/yr).

    Examples
    --------
    >>> round(taylor_rule(100), 4)
    7.9057

    References
    ----------
    .. [1] Taylor, H.K. (1977). Mine valuation and feasibility studies.
           *Mineral Industry Costs*, Northwest Mining Assoc.
    """
    if reserves_mt <= 0:
        raise ValueError("Reserves must be positive.")
    return float(0.25 * reserves_mt**0.75)
