"""Mining taxation and royalty calculations.

This module provides functions for computing royalties, tax shields,
after-tax cash flows, inflation adjustments, and capital recovery factors
used in mining project financial analysis.

References
----------
.. [1] Otto, J.M. et al. (2006). *Mining Royalties: A Global Study of
       Their Impact on Investors, Government, and Civil Society*. World Bank.
.. [2] Stermole, F.J. & Stermole, J.M. (2014). *Economic Evaluation and
       Investment Decision Methods*, 14th ed.
.. [3] Fisher, I. (1930). *The Theory of Interest*. Macmillan.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Royalty Cost
# ---------------------------------------------------------------------------


def royalty_cost(
    gross_revenue: float,
    royalty_rate: float,
) -> float:
    """Calculate royalty payable on gross revenue.

    .. math::

        R = \\text{revenue} \\times r

    Parameters
    ----------
    gross_revenue : float
        Gross revenue in currency units.  Must be non-negative.
    royalty_rate : float
        Royalty rate as a decimal fraction (e.g. 0.05 for 5 %).
        Must be in [0, 1].

    Returns
    -------
    float
        Royalty amount in currency units.

    Examples
    --------
    >>> royalty_cost(10_000_000, 0.05)
    500000.0

    References
    ----------
    .. [1] Otto et al. (2006), Ch. 3.
    """
    validate_non_negative(gross_revenue, "gross_revenue")
    validate_range(royalty_rate, 0.0, 1.0, "royalty_rate")

    return float(gross_revenue * royalty_rate)


# ---------------------------------------------------------------------------
# Income Tax Shield
# ---------------------------------------------------------------------------


def income_tax_shield(
    depreciation: float,
    tax_rate: float,
) -> float:
    """Calculate the tax shield provided by depreciation.

    .. math::

        \\text{shield} = D \\times t

    Parameters
    ----------
    depreciation : float
        Annual depreciation charge in currency units.  Must be
        non-negative.
    tax_rate : float
        Corporate income tax rate as a decimal (e.g. 0.30 for 30 %).
        Must be in [0, 1].

    Returns
    -------
    float
        Tax savings from depreciation in currency units.

    Examples
    --------
    >>> income_tax_shield(2_000_000, 0.30)
    600000.0

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 7.3.
    """
    validate_non_negative(depreciation, "depreciation")
    validate_range(tax_rate, 0.0, 1.0, "tax_rate")

    return float(depreciation * tax_rate)


# ---------------------------------------------------------------------------
# After-Tax Cash Flow
# ---------------------------------------------------------------------------


def after_tax_cashflow(
    ebitda: float,
    depreciation: float,
    tax_rate: float,
    royalty: float,
) -> dict:
    """Calculate after-tax cash flow for a mining operation.

    .. math::

        \\text{taxable} &= EBITDA - D - R

        \\text{tax} &= \\max(0,\\; \\text{taxable} \\times t)

        CF_{AT} &= EBITDA - \\text{tax} - R

    Parameters
    ----------
    ebitda : float
        Earnings before interest, taxes, depreciation, and
        amortisation in currency units.
    depreciation : float
        Depreciation charge in currency units.  Must be non-negative.
    tax_rate : float
        Income tax rate as a decimal (0--1).  Must be in [0, 1].
    royalty : float
        Royalty payment in currency units.  Must be non-negative.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"taxable_income"`` : float
        - ``"tax"`` : float
        - ``"after_tax_cashflow"`` : float
        - ``"effective_tax_rate"`` : float (0 if EBITDA <= 0)

    Examples
    --------
    >>> r = after_tax_cashflow(5_000_000, 1_000_000, 0.30, 500_000)
    >>> r["after_tax_cashflow"]
    3450000.0

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 7.4.
    """
    validate_non_negative(depreciation, "depreciation")
    validate_range(tax_rate, 0.0, 1.0, "tax_rate")
    validate_non_negative(royalty, "royalty")

    taxable_income = ebitda - depreciation - royalty
    tax = max(0.0, taxable_income * tax_rate)
    after_tax_cf = ebitda - tax - royalty

    effective_rate = tax / ebitda if ebitda > 0 else 0.0

    return {
        "taxable_income": float(taxable_income),
        "tax": float(tax),
        "after_tax_cashflow": float(after_tax_cf),
        "effective_tax_rate": float(effective_rate),
    }


# ---------------------------------------------------------------------------
# Real to Nominal Cash Flow
# ---------------------------------------------------------------------------


def real_to_nominal_cashflow(
    cashflow_real: float,
    inflation_rate: float,
    year: float,
) -> float:
    """Convert a real (constant-dollar) cash flow to nominal.

    Uses the Fisher equation:

    .. math::

        CF_{nominal} = CF_{real} \\times (1 + i)^t

    Parameters
    ----------
    cashflow_real : float
        Cash flow in real (constant) terms.
    inflation_rate : float
        Annual inflation rate as a decimal (e.g. 0.03 for 3 %).
        Must be non-negative.
    year : float
        Year number (may be fractional).  Must be non-negative.

    Returns
    -------
    float
        Cash flow in nominal terms.

    Examples
    --------
    >>> round(real_to_nominal_cashflow(1_000_000, 0.03, 5), 2)
    1159274.07

    References
    ----------
    .. [1] Fisher, I. (1930). *The Theory of Interest*. Macmillan.
    """
    validate_non_negative(inflation_rate, "inflation_rate")
    validate_non_negative(year, "year")

    return float(cashflow_real * (1.0 + inflation_rate) ** year)


# ---------------------------------------------------------------------------
# Capital Recovery Factor
# ---------------------------------------------------------------------------


def capital_recovery_factor(
    rate: float,
    n_periods: float,
) -> float:
    """Calculate the Capital Recovery Factor (CRF).

    .. math::

        CRF = \\frac{r (1+r)^n}{(1+r)^n - 1}

    Parameters
    ----------
    rate : float
        Interest rate per period as a decimal.  Must be positive.
    n_periods : float
        Number of periods.  Must be positive.

    Returns
    -------
    float
        Capital recovery factor.

    Examples
    --------
    >>> round(capital_recovery_factor(0.10, 10), 6)
    0.162745

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.3.
    """
    validate_positive(rate, "rate")
    validate_positive(n_periods, "n_periods")

    factor = (1.0 + rate) ** n_periods
    crf = rate * factor / (factor - 1.0)

    return float(crf)
