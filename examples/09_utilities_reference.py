"""Utilities reference: conversions, mineral database, grades, and statistics.

This example demonstrates the six sub-modules of ``minelab.utilities``:

1. **Unit conversions** — length, mass, volume, pressure, density, temperature.
2. **Mineral database** — look up properties, search by keyword.
3. **Grade conversions** — ppm ↔ percent, g/t ↔ oz/ton, metal content.
4. **Descriptive statistics** — summary stats and capping analysis on assay data.
5. **Validators** — input-validation helpers used throughout MineLab.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities import (
    capping_analysis,
    density_convert,
    descriptive_stats,
    get_mineral,
    get_sg,
    gpt_to_oz_per_ton,
    grade_tonnage_curve,
    length_convert,
    log_stats,
    mass_convert,
    metal_content,
    ppm_to_percent,
    pressure_convert,
    search_minerals,
    temperature_convert,
    validate_positive,
    validate_range,
    volume_convert,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Unit conversions
    # ------------------------------------------------------------------
    print("=== Unit Conversions ===")

    metres = length_convert(100, "ft", "m")
    print(f"  100 ft  = {metres:.4f} m")

    kg = mass_convert(1, "ton", "kg")
    print(f"  1 short ton = {kg:,.2f} kg")

    litres = volume_convert(1, "m3", "L")
    print(f"  1 m³   = {litres:,.0f} L")

    mpa = pressure_convert(1000, "psi", "MPa")
    print(f"  1 000 psi = {mpa:.4f} MPa")

    lb_ft3 = density_convert(2.7, "g/cm3", "lb/ft3")
    print(f"  2.7 g/cm³ = {lb_ft3:.2f} lb/ft³")

    fahrenheit = temperature_convert(100, "C", "F")
    print(f"  100 °C = {fahrenheit:.1f} °F")

    # ------------------------------------------------------------------
    # 2. Mineral database
    # ------------------------------------------------------------------
    print("\n=== Mineral Database ===")

    cpy = get_mineral("chalcopyrite")
    if cpy is not None:
        print(f"  Chalcopyrite: formula={cpy['formula']}, SG={cpy['sg']}")

    sg_qz = get_sg("quartz")
    print(f"  Quartz SG: {sg_qz}")

    cu_minerals = search_minerals("copper")
    print(f"  Minerals containing 'copper': {len(cu_minerals)} found")
    for m in cu_minerals[:3]:
        print(f"    - {m['name']} ({m['formula']}), SG={m['sg']}")

    # ------------------------------------------------------------------
    # 3. Grade conversions and metal content
    # ------------------------------------------------------------------
    print("\n=== Grade Conversions ===")

    pct = ppm_to_percent(5000)
    print(f"  5 000 ppm = {pct:.2f} %")

    oz = gpt_to_oz_per_ton(3.5)
    print(f"  3.5 g/t   = {oz:.4f} oz/ton")

    metal_t = metal_content(tonnage=1_000_000, grade=0.008, recovery=0.90)
    print(f"  1 Mt @ 0.8% Cu, 90% recovery -> {metal_t:,.0f} t metal")

    # Grade-tonnage curve for a small block model
    rng = np.random.default_rng(42)
    grades = rng.lognormal(mean=-1.0, sigma=0.6, size=200)
    tonnages = np.full(200, 5000.0)
    cutoffs = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    gt = grade_tonnage_curve(grades.tolist(), tonnages.tolist(), cutoffs)
    print("\n  Grade-Tonnage Curve:")
    print(f"  {'Cutoff':>8} {'Tonnes':>12} {'Avg Grade':>10} {'Metal':>10}")
    for _, row in gt.iterrows():
        print(
            f"  {row['cutoff']:8.2f} "
            f"{row['tonnes_above']:12,.0f} "
            f"{row['mean_grade_above']:10.4f} "
            f"{row['metal_above']:10,.1f}"
        )

    # ------------------------------------------------------------------
    # 4. Descriptive statistics and capping analysis
    # ------------------------------------------------------------------
    print("\n=== Descriptive Statistics ===")
    assays = rng.lognormal(mean=0.0, sigma=1.0, size=500).tolist()

    stats = descriptive_stats(assays)
    print(f"  Count:    {stats['count']:.0f}")
    print(f"  Mean:     {stats['mean']:.4f}")
    print(f"  Std Dev:  {stats['std']:.4f}")
    print(f"  CV:       {stats['cv']:.4f}")
    print(f"  Skewness: {stats['skew']:.4f}")
    print(f"  Min/Max:  {stats['min']:.4f} / {stats['max']:.4f}")

    lstats = log_stats(assays)
    print(f"\n  Log-Stats (mean): {lstats['mean']:.4f}")

    cap = capping_analysis(assays, percentiles=[90.0, 95.0, 97.5, 99.0])
    print("\n  Capping Analysis:")
    print(f"  {'%ile':>6} {'Cap':>8} {'Mean':>8} {'CV':>6} {'Metal Lost':>11}")
    for _, row in cap.iterrows():
        print(
            f"  {row['percentile']:6.1f} "
            f"{row['threshold']:8.3f} "
            f"{row['capped_mean']:8.4f} "
            f"{row['capped_cv']:6.3f} "
            f"{row['pct_metal_removed']:10.2f}%"
        )

    # ------------------------------------------------------------------
    # 5. Validators (error handling demo)
    # ------------------------------------------------------------------
    print("\n=== Validators ===")

    validate_positive(10.0, "drill_depth")
    print("  validate_positive(10.0, 'drill_depth') — OK")

    validate_range(65.0, 0.0, 100.0, "rmr_value")
    print("  validate_range(65.0, 0, 100, 'rmr_value') — OK")

    for bad_val, label in [(-5, "negative_depth"), (0, "zero_rate")]:
        try:
            validate_positive(bad_val, label)
        except ValueError as exc:
            print(f"  validate_positive({bad_val}, '{label}') -> ValueError: {exc}")


if __name__ == "__main__":
    main()
