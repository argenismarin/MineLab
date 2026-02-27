"""Economic evaluation: NPV, IRR, Monte Carlo, and sensitivity analysis.

This example demonstrates a complete economic evaluation workflow for a
hypothetical open-pit copper mining project using MineLab:

1. Define project parameters (capex, opex, production, price).
2. Build annual cash flows and compute NPV, IRR, and payback period.
3. Run a Monte Carlo simulation with uncertain copper price and opex.
4. Compute confidence intervals on the simulated NPV distribution.
5. Perform a tornado sensitivity analysis to rank key drivers.
6. Generate spider-plot data showing NPV sensitivity to each parameter.
"""

from __future__ import annotations

import numpy as np

from minelab.economics import (
    confidence_intervals,
    irr,
    npv,
    payback_period,
    run_monte_carlo,
    spider_plot_data,
    tornado_analysis,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Project parameters
    # ------------------------------------------------------------------
    capex = 150_000_000  # USD, initial capital expenditure
    annual_ore = 5_000_000  # tonnes per year
    mine_life = 10  # years
    cu_grade = 0.008  # 0.8% Cu head grade (fraction)
    recovery = 0.88  # 88% metallurgical recovery
    cu_price = 8_500  # USD/tonne copper
    opex_per_t = 12.0  # USD/tonne ore
    discount_rate = 0.10  # 10% real discount rate

    # Annual revenue and costs
    annual_metal = annual_ore * cu_grade * recovery  # tonnes Cu
    annual_revenue = annual_metal * cu_price
    annual_opex = annual_ore * opex_per_t
    annual_cf = annual_revenue - annual_opex

    print("=== Project Parameters ===")
    print(f"  CAPEX:            USD {capex:>14,.0f}")
    print(f"  Annual ore:       {annual_ore:>14,.0f} t/yr")
    print(f"  Cu grade:         {cu_grade * 100:>14.2f} %")
    print(f"  Recovery:         {recovery * 100:>14.1f} %")
    print(f"  Cu price:         USD {cu_price:>10,.0f} /t")
    print(f"  OPEX:             USD {opex_per_t:>10.2f} /t ore")
    print(f"  Annual metal:     {annual_metal:>14,.0f} t Cu")
    print(f"  Annual revenue:   USD {annual_revenue:>14,.0f}")
    print(f"  Annual net CF:    USD {annual_cf:>14,.0f}")

    # ------------------------------------------------------------------
    # 2. Build cash flows and compute NPV, IRR, payback
    # ------------------------------------------------------------------
    cashflows = [-capex] + [annual_cf] * mine_life

    project_npv = npv(discount_rate, cashflows)
    project_irr = irr(cashflows)
    project_payback = payback_period(cashflows)

    print("\n=== Deterministic Evaluation ===")
    print(f"  NPV (@ {discount_rate:.0%}):    USD {project_npv:>14,.0f}")
    print(f"  IRR:              {project_irr:>14.2%}")
    print(f"  Payback period:   {project_payback:>14.2f} years")

    # ------------------------------------------------------------------
    # 3. Monte Carlo simulation on NPV
    # ------------------------------------------------------------------
    # Uncertain parameters: copper price and opex vary each year
    # Year 0 is fixed CAPEX; years 1-10 have uncertain net cash flows

    def project_npv_model(cu_price_mc: float, opex_mc: float) -> float:
        """Compute NPV for a single Monte Carlo iteration."""
        metal = annual_ore * cu_grade * recovery
        revenue = metal * cu_price_mc
        cost = annual_ore * opex_mc
        cf_annual = revenue - cost
        cfs = [-capex] + [cf_annual] * mine_life
        return npv(discount_rate, cfs)

    mc_distributions = {
        "cu_price_mc": ("triangular", (6_500.0, 8_500.0, 11_000.0)),
        "opex_mc": ("triangular", (10.0, 12.0, 15.0)),
    }

    n_sims = 10_000
    mc_results = run_monte_carlo(
        project_npv_model,
        mc_distributions,
        n_sims,
        rng=np.random.default_rng(123),
    )

    ci = confidence_intervals(mc_results, levels=(10, 50, 90))

    print(f"\n=== Monte Carlo Simulation ({n_sims:,} iterations) ===")
    print(f"  Mean NPV:         USD {mc_results.mean():>14,.0f}")
    print(f"  Std Dev:          USD {mc_results.std():>14,.0f}")
    print(f"  P10 (pessimistic):USD {ci['P10']:>14,.0f}")
    print(f"  P50 (median):     USD {ci['P50']:>14,.0f}")
    print(f"  P90 (optimistic): USD {ci['P90']:>14,.0f}")
    print(f"  Prob NPV > 0:     {(mc_results > 0).mean():>14.1%}")

    # ------------------------------------------------------------------
    # 4. Tornado sensitivity analysis
    # ------------------------------------------------------------------
    base_params = {
        "cu_price_mc": cu_price,
        "opex_mc": opex_per_t,
    }
    variations = {
        "cu_price_mc": (6_500.0, 11_000.0),
        "opex_mc": (10.0, 15.0),
    }

    tornado = tornado_analysis(base_params, variations, project_npv_model)

    print("\n=== Tornado Sensitivity Analysis ===")
    print(f"  {'Parameter':<16} {'Low NPV':>16} {'High NPV':>16} {'Swing':>16}")
    for row in tornado:
        print(
            f"  {row['param']:<16} "
            f"USD {row['low']:>12,.0f} "
            f"USD {row['high']:>12,.0f} "
            f"USD {row['swing']:>12,.0f}"
        )

    # ------------------------------------------------------------------
    # 5. Spider-plot data
    # ------------------------------------------------------------------
    spider = spider_plot_data(
        base_params,
        param_names=["cu_price_mc", "opex_mc"],
        range_pct=0.25,
        steps=5,
        model_fn=project_npv_model,
    )

    print("\n=== Spider Plot Data (+/- 25%) ===")
    for param, (pcts, vals) in spider.items():
        print(f"\n  {param}:")
        for pct, val in zip(pcts, vals, strict=True):
            print(f"    {pct:+6.1f}%  ->  NPV = USD {val:>14,.0f}")


if __name__ == "__main__":
    main()
