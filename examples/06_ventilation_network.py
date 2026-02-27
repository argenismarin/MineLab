"""Example 06 -- Underground mine ventilation network design.

This script demonstrates a complete ventilation engineering workflow:
  1. Calculate airway resistances using the Atkinson equation.
  2. Determine air quantity requirements for diesel dilution and methane.
  3. Compute series and parallel equivalent resistances.
  4. Solve airflow distribution with the Hardy Cross iterative method.
  5. Select a fan operating point and compute power requirements.

All functions come from ``minelab.ventilation``.
"""

from __future__ import annotations

import numpy as np

from minelab.ventilation import (
    air_for_diesel,
    atkinson_resistance,
    fan_operating_point,
    fan_power,
    hardy_cross,
    methane_dilution,
    parallel_resistance,
    pressure_drop,
    series_resistance,
)


def main() -> None:
    print("=" * 60)
    print("UNDERGROUND MINE VENTILATION NETWORK DESIGN")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Air quantity requirements
    # ------------------------------------------------------------------
    # Diesel fleet: 3 trucks at 300 kW each + 2 loaders at 200 kW each
    total_diesel_kw = 3 * 300 + 2 * 200  # 1300 kW total
    q_diesel = air_for_diesel(total_diesel_kw, altitude=0.0)

    # Methane dilution: coal seam emitting 0.3 m3/s, keep below 1%
    q_methane = methane_dilution(emission_rate=0.3, target_conc=0.01)

    # The governing air requirement is the maximum
    q_required = max(q_diesel, q_methane)
    print("\n1. Air Quantity Requirements")
    print(f"   Diesel fleet: {total_diesel_kw} kW total")
    print(f"   Q for diesel dilution: {q_diesel:.1f} m3/s")
    print(f"   Q for methane dilution: {q_methane:.1f} m3/s")
    print(f"   Governing requirement: {q_required:.1f} m3/s")

    # ------------------------------------------------------------------
    # 2. Airway resistance calculations
    # ------------------------------------------------------------------
    # Main intake decline: 500 m long, 5x4 m cross-section
    k_shotcrete = 0.004  # Friction factor for shotcrete-lined airway
    r_intake = atkinson_resistance(
        k=k_shotcrete,
        length=500,
        perimeter=18.0,
        area=20.0,
    )

    # Main haulage level: 800 m long, 5x5 m cross-section
    k_smooth = 0.003  # Smooth concrete lined
    r_haulage = atkinson_resistance(
        k=k_smooth,
        length=800,
        perimeter=20.0,
        area=25.0,
    )

    # Return airway (exhaust raise): 300 m long, 4x4 m cross-section
    k_rough = 0.012  # Rough rock surface
    r_return = atkinson_resistance(
        k=k_rough,
        length=300,
        perimeter=16.0,
        area=16.0,
    )

    # Parallel stope airways: two identical stopes
    k_stope = 0.010  # Stope airway friction factor
    r_stope1 = atkinson_resistance(
        k=k_stope,
        length=200,
        perimeter=14.0,
        area=12.0,
    )
    r_stope2 = atkinson_resistance(
        k=k_stope,
        length=200,
        perimeter=14.0,
        area=12.0,
    )

    print("\n2. Airway Resistances (Atkinson Equation)")
    print(f"   Intake decline:  R = {r_intake:.6f} Ns2/m8")
    print(f"   Haulage level:   R = {r_haulage:.6f} Ns2/m8")
    print(f"   Return airway:   R = {r_return:.6f} Ns2/m8")
    print(f"   Stope 1:         R = {r_stope1:.6f} Ns2/m8")
    print(f"   Stope 2:         R = {r_stope2:.6f} Ns2/m8")

    # ------------------------------------------------------------------
    # 3. Equivalent resistances
    # ------------------------------------------------------------------
    # Stopes are in parallel, then in series with intake and return
    r_stopes_parallel = parallel_resistance([r_stope1, r_stope2])
    r_total_series = series_resistance([r_intake, r_haulage, r_stopes_parallel, r_return])

    print("\n3. Equivalent Resistances")
    print(f"   Stopes in parallel: R = {r_stopes_parallel:.6f} Ns2/m8")
    print(f"   Total series:       R = {r_total_series:.6f} Ns2/m8")

    # Pressure drop at the required airflow
    dp_total = pressure_drop(r_total_series, q_required)
    print(f"   Pressure drop at Q = {q_required:.0f} m3/s: {dp_total:.1f} Pa")

    # ------------------------------------------------------------------
    # 4. Hardy Cross network solution
    # ------------------------------------------------------------------
    # Simple two-branch parallel network representing the stopes
    # Total flow entering the parallel section = q_required
    # Initial guess: split evenly between two stopes
    q_half = q_required / 2.0
    branches = [
        {
            "from": 0,
            "to": 1,
            "resistance": r_stope1,
            "Q_init": q_half + 5,  # Slightly unbalanced initial guess
            "fan_pressure": 0.0,
            "mesh": 0,
        },
        {
            "from": 0,
            "to": 1,
            "resistance": r_stope2,
            "Q_init": q_half - 5,
            "fan_pressure": 0.0,
            "mesh": 0,
        },
    ]
    hc_result = hardy_cross(branches, junctions=2, tol=0.01, max_iter=100)

    print("\n4. Hardy Cross Network Solution (parallel stopes)")
    print(f"   Converged: {hc_result['converged']} in {hc_result['iterations']} iterations")
    print(f"   Stope 1 flow: {hc_result['flows'][0]:.2f} m3/s")
    print(f"   Stope 2 flow: {hc_result['flows'][1]:.2f} m3/s")
    print(f"   Stope 1 pressure drop: {hc_result['pressure_drops'][0]:.1f} Pa")
    print(f"   Stope 2 pressure drop: {hc_result['pressure_drops'][1]:.1f} Pa")

    # ------------------------------------------------------------------
    # 5. Fan selection and power
    # ------------------------------------------------------------------
    # Define a typical axial fan characteristic curve
    fan_q = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    fan_p = np.array(
        [4000, 3900, 3700, 3400, 3000, 2500, 1900, 1200, 600, 200, 0],
        dtype=float,
    )

    # Find operating point: intersection of fan curve and system curve
    op = fan_operating_point(fan_q, fan_p, system_resistance=r_total_series)

    print("\n5. Fan Selection")
    print(f"   System resistance: {r_total_series:.6f} Ns2/m8")
    print(f"   Operating airflow:  Q = {op['Q_operating']:.1f} m3/s")
    print(f"   Operating pressure: P = {op['P_operating']:.0f} Pa")

    # Check if the fan meets the required airflow
    if op["Q_operating"] >= q_required:
        print(f"   Status: Fan MEETS the requirement of {q_required:.0f} m3/s")
    else:
        deficit = q_required - op["Q_operating"]
        print(f"   Status: Fan SHORT by {deficit:.1f} m3/s -- consider larger fan")

    # Fan power calculation
    eta = 0.72  # Fan efficiency (typical for mine axial fans)
    power_w = fan_power(op["Q_operating"], op["P_operating"], eta)
    power_kw = power_w / 1000.0

    print(f"\n   Fan efficiency: {eta * 100:.0f}%")
    print(f"   Fan motor power: {power_kw:.1f} kW ({power_kw / 0.746:.0f} HP)")

    # Annual energy cost estimate
    hours_per_year = 8760
    electricity_cost = 0.08  # $/kWh
    annual_cost = power_kw * hours_per_year * electricity_cost
    print(f"   Annual energy cost: ${annual_cost:,.0f}/year (at ${electricity_cost}/kWh)")

    print(f"\n{'=' * 60}")
    print("Ventilation design complete.")


if __name__ == "__main__":
    main()
