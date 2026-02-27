"""Production management: blending, grade control, stockpiles, and reconciliation.

This example demonstrates production-management workflows for a copper mine
receiving ore from three mining fronts with different head grades:

1. **Blending** — calculate blend grade and optimise feed to meet plant targets.
2. **Grade control** — SMU classification and the information effect.
3. **Stockpile simulation** — FIFO vs LIFO reclaim strategies.
4. **Reconciliation** — F-factors and variance analysis between model and plant.
"""

from __future__ import annotations

import numpy as np

from minelab.production import (
    blend_grade,
    blend_optimize,
    f_factors,
    information_effect,
    reconciliation_report,
    smu_classification,
    stockpile_fifo,
    stockpile_lifo,
    variance_analysis,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Scenario: 3 mining fronts feeding a 10 000 t/d concentrator
    # ------------------------------------------------------------------
    fronts = {
        "North": {"tonnes": 4000, "cu_grade": 0.65},
        "South": {"tonnes": 3500, "cu_grade": 0.82},
        "East": {"tonnes": 2500, "cu_grade": 0.45},
    }

    print("=== Mining Fronts ===")
    for name, f in fronts.items():
        print(f"  {name}: {f['tonnes']:,} t/d @ {f['cu_grade']:.2f}% Cu")

    # ------------------------------------------------------------------
    # 1. Blending
    # ------------------------------------------------------------------
    print("\n=== Blending ===")

    tonnages = [f["tonnes"] for f in fronts.values()]
    grades = [f["cu_grade"] for f in fronts.values()]

    blended = blend_grade(tonnages, grades)
    print(f"  Simple blend (all fronts): {blended:.4f}% Cu")
    print(f"  Total feed: {sum(tonnages):,} t/d")

    # Optimise blend to achieve target grade window [0.55, 0.70]% Cu
    sources = [
        {"tonnage_available": 4000.0, "grades": {"Cu": 0.65}},
        {"tonnage_available": 3500.0, "grades": {"Cu": 0.82}},
        {"tonnage_available": 2500.0, "grades": {"Cu": 0.45}},
    ]
    constraints = {"Cu": {"min": 0.55, "max": 0.70}}

    opt = blend_optimize(sources, constraints, tonnage_target=8000.0)
    print("\n  Optimised blend (target 8 000 t/d, Cu 0.55-0.70%):")
    print(f"  Feasible: {opt['feasible']}")
    for name, t in zip(fronts.keys(), opt["tonnages"], strict=True):
        print(f"    {name}: {t:,.0f} t")
    print(f"  Blend Cu grade: {opt['blend_grade']['Cu']:.4f}%")

    # ------------------------------------------------------------------
    # 2. Grade control — SMU classification & information effect
    # ------------------------------------------------------------------
    print("\n=== Grade Control ===")

    rng = np.random.default_rng(42)
    true_grades = rng.lognormal(mean=-0.5, sigma=0.4, size=500)
    # Estimated grades add noise (simulating kriging estimation variance)
    estimation_noise = rng.normal(0, 0.08, size=500)
    est_grades = np.maximum(true_grades + estimation_noise, 0.01)

    cutoff = 0.50  # % Cu

    smu = smu_classification(true_grades, cutoff)
    print(f"  SMU classification (cutoff {cutoff}% Cu):")
    print(f"    Ore blocks:   {smu['ore_count']} (avg {smu['ore_grade']:.3f}%)")
    print(f"    Waste blocks: {smu['waste_count']} (avg {smu['waste_grade']:.3f}%)")

    ie = information_effect(true_grades, est_grades, cutoff)
    print("\n  Information effect:")
    print(f"    True ore blocks:  {ie['true_ore_tonnage']}")
    print(f"    Est. ore blocks:  {ie['est_ore_tonnage']}")
    print(f"    Tonnage change:   {ie['tonnage_change_pct']:+.1f}%")
    print(f"    Grade change:     {ie['grade_change_pct']:+.1f}%")

    # ------------------------------------------------------------------
    # 3. Stockpile simulation — FIFO vs LIFO
    # ------------------------------------------------------------------
    print("\n=== Stockpile Simulation ===")

    # Build up stockpile over 5 shifts, then reclaim in 3 pulls
    additions = [
        {"tonnes": 2000, "grade": 0.55},
        {"tonnes": 1500, "grade": 0.70},
        {"tonnes": 1800, "grade": 0.48},
        {"tonnes": 2200, "grade": 0.62},
        {"tonnes": 1600, "grade": 0.80},
    ]
    reclaims = [2500, 3000, 2000]

    fifo = stockpile_fifo(additions, reclaims)
    lifo = stockpile_lifo(additions, reclaims)

    print("  FIFO reclaim:")
    for i, r in enumerate(fifo["reclaimed"], 1):
        print(f"    Pull {i}: {r['tonnes']:,.0f} t @ {r['grade']:.4f}% Cu")

    print("  LIFO reclaim:")
    for i, r in enumerate(lifo["reclaimed"], 1):
        print(f"    Pull {i}: {r['tonnes']:,.0f} t @ {r['grade']:.4f}% Cu")

    fifo_remaining = sum(r["tonnes"] for r in fifo["remaining"])
    lifo_remaining = sum(r["tonnes"] for r in lifo["remaining"])
    print(f"  Remaining in stockpile: FIFO={fifo_remaining:,.0f} t, LIFO={lifo_remaining:,.0f} t")

    # ------------------------------------------------------------------
    # 4. Reconciliation — F-factors and variance analysis
    # ------------------------------------------------------------------
    print("\n=== Reconciliation ===")

    # Monthly data: model vs mine vs plant
    periods = [
        {
            "model_tonnes": 300_000,
            "model_grade": 0.65,
            "mined_tonnes": 310_000,
            "mined_grade": 0.62,
            "plant_tonnes": 305_000,
            "plant_grade": 0.63,
        },
        {
            "model_tonnes": 280_000,
            "model_grade": 0.70,
            "mined_tonnes": 275_000,
            "mined_grade": 0.68,
            "plant_tonnes": 270_000,
            "plant_grade": 0.69,
        },
        {
            "model_tonnes": 320_000,
            "model_grade": 0.60,
            "mined_tonnes": 315_000,
            "mined_grade": 0.58,
            "plant_tonnes": 310_000,
            "plant_grade": 0.59,
        },
    ]

    # Single-period F-factors
    p1 = periods[0]
    ff = f_factors(
        p1["model_tonnes"],
        p1["model_grade"],
        p1["mined_tonnes"],
        p1["mined_grade"],
        p1["plant_tonnes"],
        p1["plant_grade"],
    )
    print("  Period 1 F-factors:")
    print(
        f"    F1 (mine/model):  tonnes={ff['F1_tonnes']:.3f}, "
        f"grade={ff['F1_grade']:.3f}, metal={ff['F1_metal']:.3f}"
    )
    print(
        f"    F2 (plant/mine):  tonnes={ff['F2_tonnes']:.3f}, "
        f"grade={ff['F2_grade']:.3f}, metal={ff['F2_metal']:.3f}"
    )
    print(
        f"    F3 (plant/model): tonnes={ff['F3_tonnes']:.3f}, "
        f"grade={ff['F3_grade']:.3f}, metal={ff['F3_metal']:.3f}"
    )

    # Multi-period reconciliation report
    report = reconciliation_report(periods)
    avg = report["averages"]
    print("\n  3-Month Averages:")
    print(f"    F1 metal: {avg['F1_metal']:.3f}")
    print(f"    F2 metal: {avg['F2_metal']:.3f}")
    print(f"    F3 metal: {avg['F3_metal']:.3f}")

    # Variance analysis (planned vs actual for latest period)
    p3 = periods[2]
    va = variance_analysis(
        p3["model_tonnes"],
        p3["model_grade"],
        p3["mined_tonnes"],
        p3["mined_grade"],
    )
    print("\n  Variance Analysis (Period 3):")
    print(f"    Planned metal: {va['planned_metal']:,.1f}")
    print(f"    Actual metal:  {va['actual_metal']:,.1f}")
    print(f"    Tonnage effect:  {va['tonnage_effect']:+,.1f}")
    print(f"    Grade effect:    {va['grade_effect']:+,.1f}")
    print(f"    Combined effect: {va['combined_effect']:+,.1f}")
    print(f"    Total variance:  {va['total_variance']:+,.1f}")


if __name__ == "__main__":
    main()
