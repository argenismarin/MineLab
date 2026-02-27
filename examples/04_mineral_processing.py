"""Mineral processing: comminution, flotation, and mass balance.

This example demonstrates a simplified copper concentrator design
workflow using MineLab:

1. Compute the Bond work index from laboratory grindability data.
2. Calculate specific energy and ball mill power for the grinding circuit.
3. Model flotation kinetics using the first-order model.
4. Fit kinetics parameters from experimental time-recovery data.
5. Design a flotation bank (number of cells).
6. Evaluate a rougher-scavenger-cleaner circuit.
7. Perform a two-product mass balance (concentrate vs. tailings).
8. Verify mass balance closure.
"""

from __future__ import annotations

import numpy as np

from minelab.mineral_processing import (
    ball_mill_power,
    bond_energy,
    bond_work_index,
    check_closure,
    flotation_bank_design,
    flotation_circuit,
    flotation_first_order,
    flotation_kinetics_fit,
    two_product,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Plant design parameters
    # ------------------------------------------------------------------
    throughput = 5_000.0  # t/h (large copper concentrator)
    feed_grade_cu = 0.80  # % Cu in ROM feed
    feed_p80 = 12_000.0  # micrometers (12 mm, SAG discharge)
    product_p80 = 75.0  # micrometers (75 um, flotation feed)

    print("=== Plant Design Parameters ===")
    print(f"  Throughput:       {throughput:,.0f} t/h")
    print(f"  Feed grade:       {feed_grade_cu:.2f} % Cu")
    print(f"  Feed P80:         {feed_p80:,.0f} um ({feed_p80 / 1000:.0f} mm)")
    print(f"  Target grind P80: {product_p80:.0f} um")

    # ------------------------------------------------------------------
    # 1. Bond work index from laboratory test
    # ------------------------------------------------------------------
    # Typical grindability test results for a porphyry copper ore
    closing_screen = 106.0  # um (150 mesh)
    grindability = 1.5  # grams per revolution (Gbp)

    wi = bond_work_index(closing_screen, feed_p80, product_p80, grindability)

    print("\n=== Bond Work Index ===")
    print(f"  Closing screen:   {closing_screen:.0f} um")
    print(f"  Grindability Gbp: {grindability:.1f} g/rev")
    print(f"  Work index Wi:    {wi:.1f} kWh/t")

    # ------------------------------------------------------------------
    # 2. Specific energy and ball mill power
    # ------------------------------------------------------------------
    specific_energy = bond_energy(wi, feed_p80, product_p80)
    mill_efficiency = 0.90  # 90% mechanical efficiency
    power_kw = ball_mill_power(wi, feed_p80, product_p80, throughput, mill_efficiency)

    print("\n=== Ball Mill Power ===")
    print(f"  Specific energy:  {specific_energy:.2f} kWh/t (Bond's 3rd law)")
    print(f"  Mill efficiency:  {mill_efficiency:.0%}")
    print(f"  Required power:   {power_kw:,.0f} kW  ({power_kw / 1000:.1f} MW)")
    print(f"  Installed power:  {power_kw * 1.15 / 1000:.1f} MW (15% margin)")

    # ------------------------------------------------------------------
    # 3. Flotation kinetics (first-order model)
    # ------------------------------------------------------------------
    # Predict recovery at different residence times
    r_inf = 0.95  # ultimate recovery (fraction)
    k_rate = 0.45  # rate constant (1/min)

    print("\n=== Flotation Kinetics (First-Order) ===")
    print(f"  R_inf = {r_inf:.2f},  k = {k_rate:.2f} min^-1")
    print(f"  {'Time (min)':>10}  {'Recovery (%)':>12}")
    for t in [1, 2, 5, 10, 15, 20]:
        rec = flotation_first_order(r_inf, k_rate, float(t))
        print(f"  {t:>10d}  {rec * 100:>12.2f}")

    # ------------------------------------------------------------------
    # 4. Fit kinetics from experimental data
    # ------------------------------------------------------------------
    # Simulated lab flotation test (cumulative recovery over time)
    times = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    recoveries = np.array([0.0, 0.18, 0.32, 0.52, 0.78, 0.90, 0.93, 0.94])

    fit = flotation_kinetics_fit(times, recoveries)

    print("\n=== Kinetics Fit (lab data) ===")
    print(f"  R_inf (fitted):   {fit['r_inf']:.4f}")
    print(f"  k (fitted):       {fit['k']:.4f} min^-1")
    print(f"  R-squared:        {fit['r_squared']:.6f}")

    # ------------------------------------------------------------------
    # 5. Flotation bank design
    # ------------------------------------------------------------------
    target_recovery = 0.90
    cell_volume = 160.0  # m3 per cell (large tank cell)
    # Volumetric feed rate: throughput in m3/min (assume SG ~1.3 slurry,
    # 35% solids by weight -> ~2.5 m3/t)
    feed_rate_m3min = throughput * 2.5 / 60.0  # m3/min

    bank = flotation_bank_design(
        recovery_target=target_recovery,
        k=fit["k"],
        cell_volume=cell_volume,
        feed_rate=feed_rate_m3min,
        r_inf=fit["r_inf"],
    )

    print("\n=== Flotation Bank Design (rougher) ===")
    print(f"  Target recovery:  {target_recovery:.0%}")
    print(f"  Cell volume:      {cell_volume:.0f} m3")
    print(f"  Feed rate:        {feed_rate_m3min:.1f} m3/min")
    print(f"  Number of cells:  {bank['n_cells']}")
    print(f"  Residence/cell:   {bank['residence_time']:.2f} min")
    print(f"  Total residence:  {bank['total_residence_time']:.1f} min")

    # ------------------------------------------------------------------
    # 6. Rougher-scavenger-cleaner circuit recovery
    # ------------------------------------------------------------------
    rougher_r = 0.90
    scavenger_r = 0.50
    cleaner_r = 0.85

    circuit = flotation_circuit(rougher_r, cleaner_r, scavenger_r)

    print("\n=== Flotation Circuit Recovery ===")
    print(f"  Rougher:          {rougher_r:.0%}")
    print(f"  Scavenger:        {scavenger_r:.0%} (of rougher tails)")
    print(f"  Cleaner:          {cleaner_r:.0%}")
    print(f"  R-C recovery:     {circuit['rougher_cleaner_recovery']:.2%}")
    print(f"  Overall recovery: {circuit['overall_recovery']:.2%}")

    # ------------------------------------------------------------------
    # 7. Two-product mass balance
    # ------------------------------------------------------------------
    # Plant operating data
    feed_grade = 0.80  # % Cu in feed
    conc_grade = 28.0  # % Cu in concentrate
    tail_grade = 0.08  # % Cu in tailings

    balance = two_product(feed_grade, conc_grade, tail_grade)

    conc_tph = throughput * balance["concentrate_ratio"]
    tail_tph = throughput * balance["tailings_ratio"]

    print("\n=== Two-Product Mass Balance ===")
    print(f"  Feed:    {throughput:>8,.0f} t/h  @ {feed_grade:.2f} % Cu")
    print(f"  Conc:    {conc_tph:>8,.1f} t/h  @ {conc_grade:.1f} % Cu")
    print(f"  Tails:   {tail_tph:>8,.1f} t/h  @ {tail_grade:.2f} % Cu")
    print(f"  C/F ratio:  {balance['concentrate_ratio']:.4f}")
    print(f"  Recovery:   {balance['recovery']:.2%}")

    # ------------------------------------------------------------------
    # 8. Mass balance closure check
    # ------------------------------------------------------------------
    closure = check_closure(
        feed_mass=throughput,
        product_masses=[conc_tph, tail_tph],
        tolerance=0.02,
    )

    print("\n=== Mass Balance Closure ===")
    print(f"  Feed:             {throughput:,.0f} t/h")
    print(f"  Sum of products:  {closure['total_products']:,.1f} t/h")
    print(f"  Relative error:   {closure['error']:.4%}")
    print(f"  Closed (< 2%):    {closure['closed']}")


if __name__ == "__main__":
    main()
