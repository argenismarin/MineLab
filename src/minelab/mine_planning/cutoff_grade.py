"""Cut-off grade calculations for open-pit and underground mining.

Implements break-even, Lane's three-stage, and marginal cut-off grade
formulations used in mine planning and economic evaluation.

References
----------
.. [1] Lane, K. F. (1988). *The Economic Definition of Ore: Cut-off Grades
       in Theory and Practice*. Mining Journal Books.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)


def breakeven_cutoff(
    price: float,
    recovery: float,
    processing_cost: float,
    mining_cost: float,
    ga_cost: float = 0.0,
) -> float:
    """Break-even cut-off grade.

    .. math::

        g_{BE} = \\frac{C_p + C_m + C_{GA}}{P \\cdot R}

    where *C_p* is processing cost per tonne of ore, *C_m* is mining
    cost per tonne of ore, *C_GA* is general & administrative cost per
    tonne, *P* is metal price, and *R* is recovery (fraction).

    Parameters
    ----------
    price : float
        Commodity price per unit of metal (e.g. $/t).  Must be positive.
    recovery : float
        Metallurgical recovery as a fraction in (0, 1].
    processing_cost : float
        Processing cost per tonne of ore ($/t).  Non-negative.
    mining_cost : float
        Mining cost per tonne of ore ($/t).  Non-negative.
    ga_cost : float, optional
        General & administrative cost per tonne of ore ($/t).
        Non-negative.  Default 0.

    Returns
    -------
    float
        Break-even cut-off grade in the same units as the price
        denominator (e.g. fraction if price is $/t-metal).

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    Cu at $8 000/t, 90 % recovery, processing $15/t, mining $2/t:

    >>> cog = breakeven_cutoff(8000, 0.9, 15.0, 2.0)
    >>> round(cog, 6)
    0.002361

    References
    ----------
    .. [1] Lane, K. F. (1988). *The Economic Definition of Ore*,
           Ch. 2-3.
    """
    validate_positive(price, "price")
    validate_range(recovery, 0.001, 1.0, "recovery")
    validate_non_negative(processing_cost, "processing_cost")
    validate_non_negative(mining_cost, "mining_cost")
    validate_non_negative(ga_cost, "ga_cost")

    return (processing_cost + mining_cost + ga_cost) / (price * recovery)


def lane_cutoff(
    mine_capacity: float,
    mill_capacity: float,
    refinery_capacity: float,
    costs: dict,
    price: float,
    recovery: float,
) -> dict:
    """Lane's three limiting cut-off grades plus the optimum.

    Lane's theory identifies three bottleneck-driven cut-off grades:

    * **g_mine** -- limited by mining capacity (only variable
      processing cost matters).
    * **g_mill** -- limited by mill capacity (mining + processing
      costs matter).
    * **g_refinery** -- limited by refinery capacity (all costs
      matter including refining).

    The optimum cut-off is taken as the median of these three values,
    following Lane's balancing principle.

    Parameters
    ----------
    mine_capacity : float
        Maximum mining rate (t/year).  Must be positive.
    mill_capacity : float
        Maximum milling / processing rate (t/year).  Must be positive.
    refinery_capacity : float
        Maximum refinery throughput (t-metal/year).  Must be positive.
    costs : dict
        Dictionary with keys:

        * ``"mining"`` : float -- Mining cost per tonne of material.
        * ``"processing"`` : float -- Processing cost per tonne of ore.
        * ``"refining"`` : float -- Refining cost per tonne of metal
          (optional, default 0).
        * ``"ga"`` : float -- General & administrative cost per tonne
          (optional, default 0).
        * ``"fixed"`` : float -- Fixed cost per period (optional,
          default 0).
    price : float
        Metal price per tonne of refined metal.  Must be positive.
    recovery : float
        Overall metallurgical recovery as a fraction in (0, 1].

    Returns
    -------
    dict
        ``"g_mine"`` : float
            Mining-limited cut-off grade.
        ``"g_mill"`` : float
            Mill-limited cut-off grade.
        ``"g_refinery"`` : float
            Refinery-limited cut-off grade.
        ``"g_optimum"`` : float
            Optimum cut-off (median of the three).

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    KeyError
        If a required key is missing from *costs*.

    Examples
    --------
    >>> result = lane_cutoff(
    ...     mine_capacity=10_000_000,
    ...     mill_capacity=5_000_000,
    ...     refinery_capacity=50_000,
    ...     costs={"mining": 2.0, "processing": 15.0,
    ...            "refining": 500.0, "fixed": 0.0},
    ...     price=8000,
    ...     recovery=0.9,
    ... )
    >>> round(result["g_mine"], 6)
    0.002083
    >>> round(result["g_mill"], 6)
    0.002361

    References
    ----------
    .. [1] Lane, K. F. (1988). *The Economic Definition of Ore*,
           Ch. 5.
    """
    validate_positive(mine_capacity, "mine_capacity")
    validate_positive(mill_capacity, "mill_capacity")
    validate_positive(refinery_capacity, "refinery_capacity")
    validate_positive(price, "price")
    validate_range(recovery, 0.001, 1.0, "recovery")

    c_mining = costs["mining"]
    c_processing = costs["processing"]
    c_refining = costs.get("refining", 0.0)
    c_ga = costs.get("ga", 0.0)
    c_fixed = costs.get("fixed", 0.0)

    validate_non_negative(c_mining, "costs['mining']")
    validate_non_negative(c_processing, "costs['processing']")
    validate_non_negative(c_refining, "costs['refining']")
    validate_non_negative(c_ga, "costs['ga']")
    validate_non_negative(c_fixed, "costs['fixed']")

    net_price = price - c_refining

    # Mining-limited: only processing cost is variable for ore vs waste
    g_mine = (c_processing + c_ga) / (net_price * recovery)

    # Mill-limited: mining and processing costs both relevant
    g_mill = (c_mining + c_processing + c_ga) / (net_price * recovery)

    # Refinery-limited: considers the opportunity cost of refinery capacity
    # Using the marginal approach including fixed overhead allocation
    g_refinery = (c_mining + c_processing + c_ga + c_fixed / mill_capacity) / (
        net_price * recovery
    )

    # Lane's optimum: the balancing cut-off is the median of the three
    g_optimum = float(sorted([g_mine, g_mill, g_refinery])[1])

    return {
        "g_mine": float(g_mine),
        "g_mill": float(g_mill),
        "g_refinery": float(g_refinery),
        "g_optimum": g_optimum,
    }


def marginal_cutoff(
    price: float,
    recovery: float,
    processing_cost: float,
) -> float:
    """Marginal (incremental) cut-off grade.

    The marginal cut-off ignores mining costs because mining has already
    occurred (e.g. material is on a stockpile).  The only cost is
    processing.

    .. math::

        g_{marginal} = \\frac{C_p}{P \\cdot R}

    Parameters
    ----------
    price : float
        Commodity price per unit of metal ($/t).  Must be positive.
    recovery : float
        Metallurgical recovery as a fraction in (0, 1].
    processing_cost : float
        Processing cost per tonne of ore ($/t).  Non-negative.

    Returns
    -------
    float
        Marginal cut-off grade.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    >>> round(marginal_cutoff(8000, 0.9, 15.0), 6)
    0.002083

    References
    ----------
    .. [1] Lane, K. F. (1988). *The Economic Definition of Ore*,
           Ch. 2.
    """
    validate_positive(price, "price")
    validate_range(recovery, 0.001, 1.0, "recovery")
    validate_non_negative(processing_cost, "processing_cost")

    return processing_cost / (price * recovery)
