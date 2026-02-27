"""Project finance metrics for mining feasibility studies.

This module provides functions for debt service coverage, loan amortisation,
leverage effects, break-even analysis, and working capital estimation.

References
----------
.. [1] Gatti, S. (2013). *Project Finance in Theory and Practice*, 2nd ed.
       Academic Press.
.. [2] Stermole, F.J. & Stermole, J.M. (2014). *Economic Evaluation and
       Investment Decision Methods*, 14th ed.
.. [3] Modigliani, F. & Miller, M.H. (1958). *The Cost of Capital,
       Corporation Finance and the Theory of Investment*. AER, 48(3).
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Debt Service Coverage Ratio
# ---------------------------------------------------------------------------


def debt_service_coverage_ratio(
    ebitda: float,
    annual_debt_service: float,
) -> float:
    """Calculate the Debt Service Coverage Ratio (DSCR).

    .. math::

        DSCR = \\frac{EBITDA}{\\text{annual debt service}}

    Parameters
    ----------
    ebitda : float
        Earnings before interest, taxes, depreciation, and amortisation
        in currency units.  Must be non-negative.
    annual_debt_service : float
        Total annual debt service (principal + interest) in currency
        units.  Must be positive.

    Returns
    -------
    float
        Debt service coverage ratio (dimensionless).

    Examples
    --------
    >>> round(debt_service_coverage_ratio(15_000_000, 8_000_000), 2)
    1.88

    References
    ----------
    .. [1] Gatti (2013), Ch. 9.
    """
    validate_non_negative(ebitda, "ebitda")
    validate_positive(annual_debt_service, "annual_debt_service")

    return float(ebitda / annual_debt_service)


# ---------------------------------------------------------------------------
# Loan Amortisation
# ---------------------------------------------------------------------------


def loan_amortization(
    principal: float,
    annual_rate: float,
    n_years: int,
) -> dict:
    """Calculate loan amortisation schedule with equal annual payments.

    The annual payment is derived from the annuity formula:

    .. math::

        A = P \\, \\frac{r (1+r)^n}{(1+r)^n - 1}

    Parameters
    ----------
    principal : float
        Loan principal in currency units.  Must be positive.
    annual_rate : float
        Annual interest rate as a decimal (e.g. 0.08 for 8 %).
        Must be positive.
    n_years : int
        Loan term in years.  Must be >= 1.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"annual_payment"`` : float
        - ``"total_payment"`` : float
        - ``"total_interest"`` : float
        - ``"schedule"`` : list of dict, each with ``"year"``,
          ``"payment"``, ``"principal_portion"``,
          ``"interest_portion"``, ``"balance"``

    Examples
    --------
    >>> r = loan_amortization(1_000_000, 0.08, 5)
    >>> round(r["annual_payment"], 2)
    250456.45

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.3.
    """
    validate_positive(principal, "principal")
    validate_positive(annual_rate, "annual_rate")
    if n_years < 1:
        raise ValueError(f"'n_years' must be at least 1, got {n_years}.")

    factor = (1.0 + annual_rate) ** n_years
    annual_payment = principal * annual_rate * factor / (factor - 1.0)
    total_payment = annual_payment * n_years
    total_interest = total_payment - principal

    schedule = []
    balance = principal
    for year in range(1, n_years + 1):
        interest_portion = balance * annual_rate
        principal_portion = annual_payment - interest_portion
        balance -= principal_portion
        schedule.append(
            {
                "year": year,
                "payment": float(annual_payment),
                "principal_portion": float(principal_portion),
                "interest_portion": float(interest_portion),
                "balance": float(max(0.0, balance)),
            }
        )

    return {
        "annual_payment": float(annual_payment),
        "total_payment": float(total_payment),
        "total_interest": float(total_interest),
        "schedule": schedule,
    }


# ---------------------------------------------------------------------------
# Leverage Effect on IRR
# ---------------------------------------------------------------------------


def leverage_effect_irr(
    unlevered_irr: float,
    debt_fraction: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """Estimate the levered IRR using the Modigliani-Miller framework.

    A simplified relationship between levered and unlevered returns:

    .. math::

        IRR_L = IRR_U + (IRR_U - r_d (1-t))
                \\times \\frac{D}{1 - D}

    where *D* is the debt fraction and *r_d* the pre-tax cost of debt.

    Parameters
    ----------
    unlevered_irr : float
        Unlevered (all-equity) IRR as a decimal.  Must be non-negative.
    debt_fraction : float
        Debt fraction of total capital (0--1).  Must be in [0, 1).
    cost_of_debt : float
        Pre-tax cost of debt as a decimal.  Must be non-negative.
    tax_rate : float
        Corporate tax rate as a decimal (0--1).  Must be in [0, 1].

    Returns
    -------
    float
        Estimated levered IRR as a decimal.

    Examples
    --------
    >>> round(leverage_effect_irr(0.15, 0.60, 0.08, 0.30), 4)
    0.2516

    References
    ----------
    .. [1] Modigliani & Miller (1958).
    """
    validate_non_negative(unlevered_irr, "unlevered_irr")
    validate_range(debt_fraction, 0.0, 0.99, "debt_fraction")
    validate_non_negative(cost_of_debt, "cost_of_debt")
    validate_range(tax_rate, 0.0, 1.0, "tax_rate")

    after_tax_debt_cost = cost_of_debt * (1.0 - tax_rate)
    equity_fraction = 1.0 - debt_fraction
    leverage_ratio = debt_fraction / equity_fraction

    levered_irr = unlevered_irr + (unlevered_irr - after_tax_debt_cost) * leverage_ratio

    return float(levered_irr)


# ---------------------------------------------------------------------------
# Break-Even Metal Price
# ---------------------------------------------------------------------------


def break_even_metal_price(
    total_costs: float,
    annual_production: float,
    recovery: float,
) -> float:
    """Calculate the break-even metal price.

    .. math::

        P_{BE} = \\frac{C_{total}}{Q \\times R}

    Parameters
    ----------
    total_costs : float
        Total annual costs (OPEX + sustaining CAPEX) in currency units.
        Must be positive.
    annual_production : float
        Annual production of ore/concentrate in tonnes (or units
        consistent with cost units).  Must be positive.
    recovery : float
        Overall metallurgical recovery as a fraction (0--1).
        Must be in (0, 1].

    Returns
    -------
    float
        Break-even metal price per unit of production.

    Examples
    --------
    >>> round(break_even_metal_price(50_000_000, 100_000, 0.90), 2)
    555.56

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 3.5.
    """
    validate_positive(total_costs, "total_costs")
    validate_positive(annual_production, "annual_production")
    validate_range(recovery, 0.01, 1.0, "recovery")

    return float(total_costs / (annual_production * recovery))


# ---------------------------------------------------------------------------
# Working Capital Requirement
# ---------------------------------------------------------------------------


def working_capital_requirement(
    annual_opex: float,
    cash_cycle_days: float,
) -> float:
    """Estimate working capital requirement from cash conversion cycle.

    .. math::

        WC = \\text{OPEX} \\times \\frac{\\text{days}}{365}

    Parameters
    ----------
    annual_opex : float
        Annual operating expenditure in currency units.  Must be
        positive.
    cash_cycle_days : float
        Cash conversion cycle in days.  Must be positive.

    Returns
    -------
    float
        Required working capital in currency units.

    Examples
    --------
    >>> round(working_capital_requirement(36_500_000, 45), 2)
    4500000.0

    References
    ----------
    .. [1] Gatti (2013), Ch. 10.
    """
    validate_positive(annual_opex, "annual_opex")
    validate_positive(cash_cycle_days, "cash_cycle_days")

    return float(annual_opex * cash_cycle_days / 365.0)
