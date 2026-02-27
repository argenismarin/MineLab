"""NPV, IRR, payback, cost models, Monte Carlo, sensitivity, taxation, and finance."""

from minelab.economics.cashflow import (
    discounted_payback,
    equivalent_annual_annuity,
    irr,
    npv,
    payback_period,
    profitability_index,
)
from minelab.economics.cost_models import (
    capex_estimate,
    depreciation_declining_balance,
    depreciation_straight_line,
    opex_per_tonne,
    stripping_cost,
    taylor_rule,
)
from minelab.economics.monte_carlo import (
    confidence_intervals,
    mc_npv,
    run_monte_carlo,
    triangular_sample,
)
from minelab.economics.project_finance import (
    break_even_metal_price,
    debt_service_coverage_ratio,
    leverage_effect_irr,
    loan_amortization,
    working_capital_requirement,
)
from minelab.economics.revenue import (
    cutoff_grade_breakeven,
    gross_revenue,
    net_smelter_return,
)
from minelab.economics.sensitivity import (
    spider_plot_data,
    tornado_analysis,
)
from minelab.economics.taxation import (
    after_tax_cashflow,
    capital_recovery_factor,
    income_tax_shield,
    real_to_nominal_cashflow,
    royalty_cost,
)

__all__ = [
    # cashflow
    "npv",
    "irr",
    "payback_period",
    "discounted_payback",
    "profitability_index",
    "equivalent_annual_annuity",
    # cost_models
    "capex_estimate",
    "opex_per_tonne",
    "stripping_cost",
    "depreciation_straight_line",
    "depreciation_declining_balance",
    "taylor_rule",
    # monte_carlo
    "triangular_sample",
    "run_monte_carlo",
    "mc_npv",
    "confidence_intervals",
    # revenue
    "gross_revenue",
    "net_smelter_return",
    "cutoff_grade_breakeven",
    # sensitivity
    "tornado_analysis",
    "spider_plot_data",
    # taxation
    "royalty_cost",
    "income_tax_shield",
    "after_tax_cashflow",
    "real_to_nominal_cashflow",
    "capital_recovery_factor",
    # project_finance
    "debt_service_coverage_ratio",
    "loan_amortization",
    "leverage_effect_irr",
    "break_even_metal_price",
    "working_capital_requirement",
]
