"""Revenue calculations for mining projects.

This module provides functions for gross revenue, net smelter return (NSR),
and break-even cut-off grade estimation.

References
----------
.. [1] Hustrulid, W. et al. (2013). *Open Pit Mine Planning and Design*,
       3rd ed. CRC Press, Ch. 3.
.. [2] Lane, K.F. (1988). *The Economic Definition of Ore*. Mining Journal
       Books.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Gross Revenue
# ---------------------------------------------------------------------------


def gross_revenue(
    tonnage: float,
    grade: float,
    price: float,
    recovery: float = 1.0,
) -> float:
    """Compute gross revenue from ore sales.

    revenue = tonnage * grade * price * recovery

    Parameters
    ----------
    tonnage : float
        Ore tonnage (e.g. tonnes).  Must be >= 0.
    grade : float
        Head grade in consistent units (e.g. g/t for gold, % for base
        metals expressed as a fraction).  Must be >= 0.
    price : float
        Commodity price per unit of metal (e.g. USD/g, USD/lb).
        Must be >= 0.
    recovery : float, optional
        Metallurgical recovery as a fraction in [0, 1] (default 1.0).

    Returns
    -------
    float
        Gross revenue in the same currency as *price*.

    Raises
    ------
    ValueError
        If any input is negative or recovery is outside [0, 1].

    Examples
    --------
    >>> gross_revenue(1_000_000, 1.5, 60.0, 0.90)
    81000000.0

    References
    ----------
    .. [1] Hustrulid et al. (2013), Eq. 3-1.
    """
    if tonnage < 0:
        raise ValueError("Tonnage must be non-negative.")
    if grade < 0:
        raise ValueError("Grade must be non-negative.")
    if price < 0:
        raise ValueError("Price must be non-negative.")
    if not 0 <= recovery <= 1:
        raise ValueError("Recovery must be between 0 and 1.")
    return float(tonnage * grade * price * recovery)


# ---------------------------------------------------------------------------
# Net Smelter Return
# ---------------------------------------------------------------------------


def net_smelter_return(
    gross_rev: float,
    tc: float,
    rc: float,
    penalties: float = 0.0,
    payable_pct: float = 1.0,
) -> float:
    """Compute the Net Smelter Return (NSR).

    NSR = gross_rev * payable_pct - tc - rc - penalties

    Parameters
    ----------
    gross_rev : float
        Gross revenue (e.g. from :func:`gross_revenue`).  Must be >= 0.
    tc : float
        Treatment charge (USD).  Must be >= 0.
    rc : float
        Refining charge (USD).  Must be >= 0.
    penalties : float, optional
        Penalty deductions (USD, default 0).  Must be >= 0.
    payable_pct : float, optional
        Fraction of contained metal that is payable (default 1.0).
        Must be in [0, 1].

    Returns
    -------
    float
        Net smelter return (USD).

    Raises
    ------
    ValueError
        If inputs are out of valid range.

    Examples
    --------
    >>> net_smelter_return(1_000_000, 50_000, 20_000, penalties=5_000, payable_pct=0.95)
    875000.0

    References
    ----------
    .. [1] Hustrulid et al. (2013), Sec. 3.2.
    """
    if gross_rev < 0:
        raise ValueError("Gross revenue must be non-negative.")
    if tc < 0 or rc < 0 or penalties < 0:
        raise ValueError("Charges and penalties must be non-negative.")
    if not 0 <= payable_pct <= 1:
        raise ValueError("payable_pct must be between 0 and 1.")
    return float(gross_rev * payable_pct - tc - rc - penalties)


# ---------------------------------------------------------------------------
# Break-Even Cut-Off Grade
# ---------------------------------------------------------------------------


def cutoff_grade_breakeven(
    price: float,
    recovery: float,
    cost_per_tonne: float,
) -> float:
    """Compute the break-even cut-off grade (COG).

    COG = cost_per_tonne / (price * recovery)

    The result is in the same unit system as the grade implied by *price*.
    For example, if *price* is USD per gram and grades are in g/t, then
    COG is in g/t.

    Parameters
    ----------
    price : float
        Commodity price per unit of metal.  Must be > 0.
    recovery : float
        Metallurgical recovery as a fraction in (0, 1].
    cost_per_tonne : float
        Total cost per tonne of ore (mining + processing + G&A).
        Must be >= 0.

    Returns
    -------
    float
        Break-even cut-off grade.

    Raises
    ------
    ValueError
        If *price* <= 0, *recovery* not in (0, 1], or *cost_per_tonne* < 0.

    Examples
    --------
    >>> cutoff_grade_breakeven(60.0, 0.90, 30.0)
    0.5555...

    References
    ----------
    .. [1] Lane, K.F. (1988). *The Economic Definition of Ore*, Ch. 2.
    """
    if price <= 0:
        raise ValueError("Price must be positive.")
    if not 0 < recovery <= 1:
        raise ValueError("Recovery must be in (0, 1].")
    if cost_per_tonne < 0:
        raise ValueError("Cost per tonne must be non-negative.")
    return float(cost_per_tonne / (price * recovery))
