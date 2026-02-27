"""Example 05 -- Slope stability analysis for an open-pit mine.

This script demonstrates a complete geomechanical workflow:
  1. Classify rock mass using RMR (Bieniawski 1989).
  2. Convert RMR to GSI for the Hoek-Brown failure criterion.
  3. Evaluate intact rock and rock mass strength envelopes.
  4. Derive equivalent Mohr-Coulomb parameters.
  5. Run Bishop's simplified slope stability analysis.
  6. Estimate support requirements (rock bolt design, shotcrete, stand-up time).

All functions come from ``minelab.geomechanics``.
"""

from __future__ import annotations

import numpy as np

from minelab.geomechanics import (
    bishop_simplified,
    gsi_from_rmr,
    hoek_brown_intact,
    hoek_brown_parameters,
    hoek_brown_rock_mass,
    mohr_coulomb_fit,
    rmr_bieniawski,
    rock_bolt_design,
    shotcrete_thickness,
    stand_up_time,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Rock Mass Rating (RMR89) classification
    # ------------------------------------------------------------------
    # Input ratings for a copper porphyry deposit (typical bench face)
    rmr_result = rmr_bieniawski(
        ucs_rating=12,  # UCS ~100 MPa  (rating 12/15)
        rqd_rating=15,  # RQD ~75%      (rating 15/20)
        spacing_rating=12,  # Spacing 0.3-1m (rating 12/20)
        condition_rating=20,  # Slightly rough, no infill (rating 20/30)
        groundwater_rating=10,  # Damp conditions (rating 10/15)
        orientation_adj=-5,  # Slightly unfavorable orientation
    )
    rmr_value = rmr_result["rmr"]
    print("=" * 60)
    print("SLOPE STABILITY ANALYSIS -- OPEN PIT MINE")
    print("=" * 60)
    print("\n1. RMR89 Classification")
    print(f"   RMR = {rmr_value}")
    print(f"   Class: {rmr_result['class_number']} - {rmr_result['description']}")

    # ------------------------------------------------------------------
    # 2. GSI estimation from RMR
    # ------------------------------------------------------------------
    gsi = gsi_from_rmr(rmr_value)
    print("\n2. Geological Strength Index")
    print(f"   GSI = {gsi:.1f}  (from RMR89 - 5)")

    # ------------------------------------------------------------------
    # 3. Hoek-Brown failure criterion
    # ------------------------------------------------------------------
    # Intact rock properties
    sigci = 120.0  # UCS of intact rock (MPa)
    mi = 25.0  # Material constant for granodiorite
    d_factor = 0.7  # Disturbance factor for production blasting

    # Intact rock strength at various confining pressures
    sigma3_values = np.array([0, 2, 5, 10, 20])
    sigma1_intact = hoek_brown_intact(sigma3_values, sigci, mi)
    print("\n3. Hoek-Brown Failure Criterion")
    print(f"   Intact rock: sigci = {sigci} MPa, mi = {mi}")
    print(f"   sigma3 (MPa) : {sigma3_values.tolist()}")
    print(f"   sigma1 intact: {[round(s, 1) for s in sigma1_intact]}")

    # Hoek-Brown parameters for rock mass
    hb_params = hoek_brown_parameters(gsi, mi, d_factor)
    print(f"\n   Rock mass HB parameters (GSI={gsi}, D={d_factor}):")
    print(f"   mb = {hb_params['mb']:.4f}")
    print(f"   s  = {hb_params['s']:.6f}")
    print(f"   a  = {hb_params['a']:.4f}")

    # Rock mass strength envelope
    sigma1_mass = hoek_brown_rock_mass(sigma3_values, sigci, gsi, mi, d_factor)
    print(f"   sigma1 rock mass: {[round(s, 1) for s in sigma1_mass]}")

    # ------------------------------------------------------------------
    # 4. Equivalent Mohr-Coulomb parameters
    # ------------------------------------------------------------------
    mc = mohr_coulomb_fit(sigci, gsi, mi, d_factor)
    print("\n4. Equivalent Mohr-Coulomb Parameters")
    print(f"   Cohesion       = {mc['cohesion']:.2f} MPa")
    print(f"   Friction angle = {mc['friction_angle']:.1f} degrees")

    # ------------------------------------------------------------------
    # 5. Bishop simplified slope stability analysis
    # ------------------------------------------------------------------
    # Define 6 slices for a typical open-pit bench (12 m height, 65 deg face)
    c_kpa = mc["cohesion"] * 1000  # Convert MPa to kPa
    phi_deg = mc["friction_angle"]
    gamma = 26.0  # Unit weight of rock mass (kN/m3)

    slices = []
    base_angles = [5, 15, 30, 45, 55, 40]  # Base inclinations (degrees)
    widths = [3.0, 3.0, 2.5, 2.5, 2.0, 2.0]  # Slice widths (m)
    heights = [2.0, 4.0, 7.0, 10.0, 11.0, 8.0]  # Average slice heights (m)

    for w, h, alpha in zip(widths, heights, base_angles, strict=True):
        weight = gamma * w * h  # Simplified weight (kN/m)
        slices.append(
            {
                "width": w,
                "weight": weight,
                "base_angle": alpha,
                "cohesion": c_kpa,
                "friction_angle": phi_deg,
                "pore_pressure": 0.0,  # Dry slope assumption
            }
        )

    radius = 25.0  # Circular failure surface radius (m)
    bishop_result = bishop_simplified(slices, radius)
    print("\n5. Bishop Simplified Slope Stability")
    print(f"   Number of slices: {len(slices)}")
    print(f"   Failure surface radius: {radius} m")
    print(f"   Factor of Safety (FOS): {bishop_result['fos']:.3f}")
    print(f"   Converged: {bishop_result['converged']}")
    print(f"   Iterations: {bishop_result['iterations']}")

    if bishop_result["fos"] >= 1.3:
        assessment = "STABLE (FOS >= 1.3)"
    elif bishop_result["fos"] >= 1.0:
        assessment = "MARGINALLY STABLE (1.0 <= FOS < 1.3)"
    else:
        assessment = "UNSTABLE (FOS < 1.0)"
    print(f"   Assessment: {assessment}")

    # ------------------------------------------------------------------
    # 6. Support design recommendations
    # ------------------------------------------------------------------
    # Rock bolt design using Q-system equivalent
    q_equiv = 10.0 ** ((rmr_value - 44) / 9)  # Approximate Q from RMR
    span = 6.0  # Excavation span for access ramp (m)
    bolts = rock_bolt_design(q_equiv, span, esr=1.0)
    print("\n6. Support Design Recommendations")
    print(f"   Q-system (approx): {q_equiv:.2f}")
    print(f"   Rock bolt length: {bolts['bolt_length']:.1f} m")
    print(f"   Rock bolt spacing: {bolts['spacing']:.2f} m")

    # Shotcrete thickness recommendation
    screte = shotcrete_thickness(rmr_value, span)
    print(f"   Shotcrete: {screte['thickness']:.0f} mm")
    print(f"   Recommendation: {screte['recommendation']}")

    # Stand-up time estimation
    sut = stand_up_time(rmr_value, span)
    print(f"   Stand-up time: {sut['time_hours']:.1f} hours ({sut['time_days']:.1f} days)")

    print(f"\n{'=' * 60}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
