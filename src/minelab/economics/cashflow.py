"""Time value of money calculations for mining project evaluation.

This module provides core discounted cash flow (DCF) functions commonly used
in mining project feasibility studies: NPV, IRR, payback period, profitability
index, and equivalent annual annuity.

References
----------
.. [1] Stermole, F.J. & Stermole, J.M. (2014). *Economic Evaluation and
       Investment Decision Methods*, 14th ed. Investment Evaluations Corp.
.. [2] Hustrulid, W., Kuchta, M. & Martin, R. (2013). *Open Pit Mine Planning
       and Design*, 3rd ed. CRC Press, Ch. 3.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Net Present Value
# ---------------------------------------------------------------------------


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """Compute the Net Present Value of a series of cash flows.

    Parameters
    ----------
    rate : float
        Discount rate per period (e.g. 0.10 for 10 %).  Must be > -1.
    cashflows : Sequence[float]
        Cash flows starting at time 0.  Negative values represent costs.

    Returns
    -------
    float
        Net present value in the same monetary unit as the cash flows.

    Examples
    --------
    >>> npv(0.10, [-1000, 300, 420, 680])  # doctest: +ELLIPSIS
    130.72...

    References
    ----------
    .. [1] Stermole & Stermole (2014), Eq. 2-1.
    """
    if rate <= -1:
        raise ValueError("Discount rate must be greater than -1.")
    cfs = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cfs))
    return float(np.sum(cfs / (1.0 + rate) ** t))


# ---------------------------------------------------------------------------
# Internal Rate of Return
# ---------------------------------------------------------------------------


def irr(cashflows: Sequence[float], lo: float = -0.5, hi: float = 10.0) -> float:
    """Compute the Internal Rate of Return via Brent's method.

    The IRR is the discount rate at which NPV equals zero.

    Parameters
    ----------
    cashflows : Sequence[float]
        Cash flows starting at time 0.  Must contain at least one sign change.
    lo : float, optional
        Lower bound for the search interval (default -0.5).
    hi : float, optional
        Upper bound for the search interval (default 10.0, i.e. 1000 %).

    Returns
    -------
    float
        Internal rate of return as a decimal (e.g. 0.15 for 15 %).

    Raises
    ------
    ValueError
        If no sign change is detected in the cash-flow series.

    Examples
    --------
    >>> round(irr([-1000, 300, 420, 680]), 4)
    0.1634

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.6.
    """
    cfs = np.asarray(cashflows, dtype=float)
    signs = np.sign(cfs[cfs != 0])
    if np.all(signs == signs[0]):
        raise ValueError("Cash flows must contain at least one sign change to compute IRR.")

    def _npv_func(r: float) -> float:
        t = np.arange(len(cfs))
        return float(np.sum(cfs / (1.0 + r) ** t))

    return float(brentq(_npv_func, lo, hi))


# ---------------------------------------------------------------------------
# Simple Payback Period
# ---------------------------------------------------------------------------


def payback_period(cashflows: Sequence[float]) -> float:
    """Compute the simple (undiscounted) payback period.

    The payback period is the earliest time at which cumulative cash flow
    becomes non-negative.  A fractional period is returned by linear
    interpolation within the crossover period.

    Parameters
    ----------
    cashflows : Sequence[float]
        Cash flows starting at time 0.

    Returns
    -------
    float
        Payback period in the same time units as the cash-flow periods.
        Returns ``float('inf')`` if the investment is never recovered.

    Examples
    --------
    >>> payback_period([-1000, 300, 420, 680])
    2.411...
    """
    cfs = np.asarray(cashflows, dtype=float)
    cumulative = np.cumsum(cfs)

    for t in range(len(cumulative)):
        if cumulative[t] >= 0:
            if t == 0:
                return 0.0
            # Linear interpolation within the crossover period
            prev = cumulative[t - 1]
            frac = -prev / (cumulative[t] - prev)
            return float(t - 1 + frac)

    return float("inf")


# ---------------------------------------------------------------------------
# Discounted Payback Period
# ---------------------------------------------------------------------------


def discounted_payback(rate: float, cashflows: Sequence[float]) -> float:
    """Compute the discounted payback period.

    Parameters
    ----------
    rate : float
        Discount rate per period.  Must be > -1.
    cashflows : Sequence[float]
        Cash flows starting at time 0.

    Returns
    -------
    float
        Discounted payback period.  Returns ``float('inf')`` if the
        discounted investment is never recovered.

    Examples
    --------
    >>> round(discounted_payback(0.10, [-1000, 300, 420, 680]), 2)
    2.74

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.9.
    """
    if rate <= -1:
        raise ValueError("Discount rate must be greater than -1.")
    cfs = np.asarray(cashflows, dtype=float)
    t_arr = np.arange(len(cfs))
    discounted = cfs / (1.0 + rate) ** t_arr
    cumulative = np.cumsum(discounted)

    for t in range(len(cumulative)):
        if cumulative[t] >= 0:
            if t == 0:
                return 0.0
            prev = cumulative[t - 1]
            frac = -prev / (cumulative[t] - prev)
            return float(t - 1 + frac)

    return float("inf")


# ---------------------------------------------------------------------------
# Profitability Index
# ---------------------------------------------------------------------------


def profitability_index(rate: float, cashflows: Sequence[float]) -> float:
    """Compute the Profitability Index (PI).

    PI = PV(future cash flows) / |initial investment|

    Parameters
    ----------
    rate : float
        Discount rate per period.  Must be > -1.
    cashflows : Sequence[float]
        Cash flows starting at time 0.  ``cashflows[0]`` is treated as
        the initial investment (typically negative).

    Returns
    -------
    float
        Profitability index.

    Raises
    ------
    ValueError
        If the initial cash flow is zero.

    Examples
    --------
    >>> round(profitability_index(0.10, [-1000, 300, 420, 680]), 4)
    1.1307

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.10.
    """
    if rate <= -1:
        raise ValueError("Discount rate must be greater than -1.")
    cfs = np.asarray(cashflows, dtype=float)
    if cfs[0] == 0:
        raise ValueError("Initial cash flow (investment) must not be zero.")

    t = np.arange(1, len(cfs))
    pv_future = float(np.sum(cfs[1:] / (1.0 + rate) ** t))
    return pv_future / abs(cfs[0])


# ---------------------------------------------------------------------------
# Equivalent Annual Annuity
# ---------------------------------------------------------------------------


def equivalent_annual_annuity(rate: float, npv_value: float, n_years: int) -> float:
    """Compute the Equivalent Annual Annuity (EAA).

    Converts a project NPV into an equivalent constant annual amount.

    EAA = NPV * r / (1 - (1 + r)^(-n))

    Parameters
    ----------
    rate : float
        Discount rate per period.  Must be > 0.
    npv_value : float
        Net present value of the project.
    n_years : int
        Project life in years.  Must be >= 1.

    Returns
    -------
    float
        Equivalent annual annuity.

    Raises
    ------
    ValueError
        If *rate* <= 0 or *n_years* < 1.

    Examples
    --------
    >>> round(equivalent_annual_annuity(0.10, 178.77, 3), 2)
    71.89

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.12.
    """
    if rate <= 0:
        raise ValueError("Rate must be positive for EAA calculation.")
    if n_years < 1:
        raise ValueError("Project life must be at least 1 year.")
    return float(npv_value * rate / (1.0 - (1.0 + rate) ** (-n_years)))
