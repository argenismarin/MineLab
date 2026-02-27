"""Blast design: pattern geometry, fragmentation, vibration, and safety.

This example demonstrates a complete blast design workflow for a 10 m
bench in an open-pit copper mine using MineLab:

1. Compute the Lilly Blastability Index and Cunningham rock factor.
2. Calculate burden (Konya formula), spacing, stemming, and subdrill.
3. Use the all-in-one pattern_design function for comparison.
4. Compute powder factor from first principles.
5. Predict fragmentation using the Kuz-Ram model.
6. Calculate the Cunningham uniformity index.
7. Predict peak particle velocity at a nearby structure.
8. Check vibration compliance against the OSMRE standard.
9. Estimate flyrock range and minimum safety distance.
"""

from __future__ import annotations

import math

from minelab.drilling_blasting import (
    burden_konya,
    flyrock_range,
    kuz_ram,
    lilly_blastability_index,
    pattern_design,
    powder_factor,
    ppv_scaled_distance,
    rock_factor_from_bi,
    safety_distance,
    spacing_from_burden,
    stemming_length,
    subgrade_drilling,
    uniformity_index,
    vibration_compliance,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Rock and explosive properties
    # ------------------------------------------------------------------
    hole_diameter = 165.0  # mm (6.5 inch production hole)
    rho_e = 1.15  # explosive density, g/cm3 (bulk emulsion)
    rho_r = 2.65  # rock density, g/cm3 (andesite)
    bench_height = 10.0  # m
    swe = 100.0  # relative weight strength (ANFO equivalent)

    print("=== Input Parameters ===")
    print(f"  Hole diameter:    {hole_diameter:.0f} mm")
    print(f"  Explosive density:{rho_e:.2f} g/cm3 (bulk emulsion)")
    print(f"  Rock density:     {rho_r:.2f} g/cm3 (andesite)")
    print(f"  Bench height:     {bench_height:.1f} m")

    # ------------------------------------------------------------------
    # 1. Blastability index and rock factor
    # ------------------------------------------------------------------
    # Lilly sub-ratings for moderately hard, blocky andesite
    rmd = 20.0  # blocky rock mass
    jf = 30.0  # moderate joint factor
    jps = 20.0  # intermediate joint spacing
    rdi = 25.0  # density influence for SG ~2.65
    hf = 10.0  # moderate hardness

    bi = lilly_blastability_index(rmd, jf, jps, rdi, hf)
    rock_a = rock_factor_from_bi(bi)

    print("\n=== Blastability Assessment ===")
    print(f"  Lilly BI:         {bi:.1f}")
    print(f"  Rock factor A:    {rock_a:.2f}")

    # ------------------------------------------------------------------
    # 2. Individual pattern geometry calculations
    # ------------------------------------------------------------------
    b = burden_konya(hole_diameter, rho_e, rho_r)
    s = spacing_from_burden(b, ratio=1.15)
    t = stemming_length(b)
    j = subgrade_drilling(b)
    charge_len = bench_height + j - t

    print("\n=== Pattern Geometry (step-by-step) ===")
    print(f"  Burden (Konya):   {b:.2f} m")
    print(f"  Spacing (1.15xB): {s:.2f} m")
    print(f"  Stemming (0.7xB): {t:.2f} m")
    print(f"  Subdrill (0.3xB): {j:.2f} m")
    print(f"  Charge length:    {charge_len:.2f} m")
    print(f"  Hole depth:       {bench_height + j:.2f} m")

    # ------------------------------------------------------------------
    # 3. All-in-one pattern design
    # ------------------------------------------------------------------
    design = pattern_design(hole_diameter, rho_e, rho_r, bench_height)

    print("\n=== Pattern Design (all-in-one) ===")
    for key, val in design.items():
        print(f"  {key:<18s}: {val:.3f}")

    # ------------------------------------------------------------------
    # 4. Powder factor
    # ------------------------------------------------------------------
    rho_e_kgm3 = rho_e * 1000.0  # convert g/cm3 to kg/m3
    pf = powder_factor(rho_e_kgm3, hole_diameter, b, s, bench_height, t, j)

    # Charge per hole (kg)
    radius_m = hole_diameter / 2000.0
    charge_kg = rho_e_kgm3 * math.pi * radius_m**2 * charge_len

    print("\n=== Explosives Loading ===")
    print(f"  Powder factor:    {pf:.3f} kg/m3")
    print(f"  Charge per hole:  {charge_kg:.1f} kg")
    print(f"  Volume per hole:  {b * s * bench_height:.1f} m3")

    # ------------------------------------------------------------------
    # 5. Fragmentation prediction (Kuz-Ram)
    # ------------------------------------------------------------------
    frag = kuz_ram(
        powder_factor=pf,
        charge_per_hole=charge_kg,
        swe=swe,
        rock_factor=rock_a,
        n_rows=3,
    )

    print("\n=== Fragmentation (Kuz-Ram, 3 rows) ===")
    print(f"  X50 (mean size):  {frag['x50'] * 100:.1f} cm  ({frag['x50']:.3f} m)")

    # ------------------------------------------------------------------
    # 6. Uniformity index
    # ------------------------------------------------------------------
    # Bottom charge = 30% of total charge length (typical)
    bottom_charge_len = 0.30 * charge_len
    drill_accuracy = 0.15  # standard deviation, metres

    n_uni = uniformity_index(
        diameter=hole_diameter,
        burden=b,
        spacing=s,
        bench_height=bench_height,
        drill_accuracy=drill_accuracy,
        charge_length=charge_len,
        bottom_charge_length=bottom_charge_len,
    )

    print("\n=== Uniformity Index ===")
    print(f"  n (Cunningham):   {n_uni:.2f}")
    print(f"  Interpretation:   {'Well graded' if n_uni > 1.5 else 'Moderately uniform'}")

    # ------------------------------------------------------------------
    # 7. Vibration prediction
    # ------------------------------------------------------------------
    # Nearest structure at 500 m from the blast
    structure_dist = 500.0  # m
    k_site = 1140.0  # site-specific constant (hard rock)
    beta = 1.6  # attenuation exponent

    vib = ppv_scaled_distance(
        k_site=k_site,
        charge=charge_kg,
        distance=structure_dist,
        beta=beta,
    )

    print("\n=== Vibration Prediction ===")
    print(f"  Distance:         {structure_dist:.0f} m")
    print(f"  Max charge/delay: {charge_kg:.1f} kg")
    print(f"  Scaled distance:  {vib['scaled_distance']:.2f} m/kg^0.5")
    print(f"  Predicted PPV:    {vib['ppv']:.2f} mm/s")

    # ------------------------------------------------------------------
    # 8. Vibration compliance
    # ------------------------------------------------------------------
    comp = vibration_compliance(vib["ppv"], frequency=15.0, standard="OSMRE")

    print("\n=== Vibration Compliance (OSMRE) ===")
    print(f"  PPV:              {comp['ppv']:.2f} mm/s")
    print(f"  Limit:            {comp['limit']:.1f} mm/s")
    print(f"  Frequency:        {comp['frequency']:.1f} Hz")
    status = "COMPLIANT" if comp["compliant"] else "NON-COMPLIANT"
    print(f"  Status:           {status}")

    # ------------------------------------------------------------------
    # 9. Flyrock and safety distance
    # ------------------------------------------------------------------
    # Linear charge concentration (kg/m)
    charge_conc = charge_kg / charge_len

    fr = flyrock_range(hole_diameter, b, t, charge_conc)
    sd = safety_distance(fr, factor=1.5)

    print("\n=== Flyrock & Safety ===")
    print(f"  Charge conc.:     {charge_conc:.2f} kg/m")
    print(f"  Flyrock range:    {fr:.1f} m")
    print(f"  Safety distance:  {sd:.1f} m  (factor = 1.5)")


if __name__ == "__main__":
    main()
