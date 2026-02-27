"""Example 08 -- Complete mine workflow from drillholes to economics.

This script demonstrates an end-to-end mining project evaluation pipeline:
  1. Drillhole data management: build synthetic drillholes and composite.
  2. Geostatistics: fit a variogram model and estimate block grades.
  3. Resource classification: classify blocks by kriging variance.
  4. Resource reporting: generate a mineral resource statement.
  5. Mine planning: compute economic block values and cut-off grade.
  6. Economic evaluation: NPV and IRR of the project.
  7. Environmental screening: acid drainage assessment.

Functions span six modules: data_management, geostatistics,
resource_classification, mine_planning, economics, and environmental.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from minelab.data_management import composite_by_length
from minelab.economics import irr, npv
from minelab.environmental import acid_neutralizing_capacity as anc_func
from minelab.environmental import maximum_potential_acidity, napp
from minelab.geostatistics import (
    BlockModel,
    fit_variogram_manual,
    ordinary_kriging,
)
from minelab.mine_planning import block_economic_value, breakeven_cutoff
from minelab.resource_classification import (
    classify_by_kriging_variance,
    resource_statement,
)


def main() -> None:
    print("=" * 60)
    print("COMPLETE MINE WORKFLOW -- DRILLHOLES TO ECONOMICS")
    print("=" * 60)
    np.random.seed(123)

    # ==================================================================
    # STEP 1: Drillhole data and compositing
    # ==================================================================
    # Create synthetic drillhole assay data (5 vertical holes, 2m intervals)
    holes, froms, tos, cu_grades = [], [], [], []
    for h in range(5):
        hole_id = f"DH-{h + 1:03d}"
        for d in range(0, 60, 2):
            holes.append(hole_id)
            froms.append(float(d))
            tos.append(float(d + 2))
            # Simulated grade: higher in center holes and at depth
            base = 0.4 + 0.15 * (2 - abs(h - 2)) + 0.005 * d
            cu_grades.append(max(0, base + np.random.normal(0, 0.08)))

    assays_df = pd.DataFrame(
        {
            "hole_id": holes,
            "from_depth": froms,
            "to_depth": tos,
            "Cu_pct": cu_grades,
        }
    )

    # Composite to 10m intervals for geostatistical analysis
    composites = composite_by_length(assays_df, length=10.0)

    print("\n1. Drillhole Data")
    print(f"   Holes: {assays_df['hole_id'].nunique()}")
    print(f"   Raw samples: {len(assays_df)}")
    print(f"   Composites (10m): {len(composites)}")
    print(f"   Mean Cu grade: {composites['Cu_pct'].mean():.3f}%")

    # ==================================================================
    # STEP 2: Geostatistics -- variogram and kriging
    # ==================================================================
    # Assign spatial coordinates to composites (simple grid layout)
    coords_list = []
    for h in range(5):
        hole_x = 50.0 + h * 25.0  # 25m hole spacing
        hole_y = 50.0
        n_comp = len(composites[composites["hole_id"] == f"DH-{h + 1:03d}"])
        for c in range(n_comp):
            mid_depth = (c + 0.5) * 10.0
            coords_list.append([hole_x, hole_y, 100.0 - mid_depth])

    sample_coords = np.array(coords_list)
    sample_values = composites["Cu_pct"].values

    # Define a variogram model (manually fitted for this example)
    vario_model = fit_variogram_manual(
        model_type="spherical",
        nugget=0.002,
        sill=0.015,
        range_a=80.0,
    )

    # Create a block model: 10x1x6 blocks covering the drillhole area
    bm = BlockModel(
        origin=[25, 25, 40],
        block_size=[25, 50, 10],
        n_blocks=[6, 1, 6],
    )
    block_centers = bm.block_centers()

    # Ordinary Kriging estimation
    estimates, variances = ordinary_kriging(
        coords=sample_coords,
        values=sample_values,
        target_coords=block_centers,
        variogram_model=vario_model,
    )

    print("\n2. Geostatistical Estimation")
    print(
        f"   Variogram: spherical, nugget={vario_model.nugget}, "
        f"sill={vario_model.sill}, range={vario_model.range_a}m"
    )
    print(f"   Block model: {bm.n_blocks} blocks ({bm.n_total} total)")
    print(f"   Mean estimated grade: {np.mean(estimates):.3f}% Cu")
    print(f"   Kriging variance range: {np.min(variances):.4f} to {np.max(variances):.4f}")

    # ==================================================================
    # STEP 3: Resource classification by kriging variance
    # ==================================================================
    thresholds = {"measured": 0.005, "indicated": 0.010}
    classification = classify_by_kriging_variance(variances, thresholds)

    n_meas = np.sum(classification == 1)
    n_ind = np.sum(classification == 2)
    n_inf = np.sum(classification == 3)
    print("\n3. Resource Classification")
    print(f"   Measured:  {n_meas} blocks")
    print(f"   Indicated: {n_ind} blocks")
    print(f"   Inferred:  {n_inf} blocks")

    # ==================================================================
    # STEP 4: Resource statement
    # ==================================================================
    density = 2.7  # t/m3
    block_volume = float(np.prod(bm.block_size))  # m3 per block
    block_tonnes = np.full(bm.n_total, density * block_volume)
    cutoff_grade = 0.25  # 0.25% Cu cut-off

    statement = resource_statement(
        block_tonnages=block_tonnes,
        block_grades=estimates,
        classification=classification,
        cutoff=cutoff_grade,
        density=density,
    )

    print(f"\n4. Mineral Resource Statement (COG >= {cutoff_grade}% Cu)")
    print(f"   {'Category':<12} {'Tonnes':>12} {'Grade (%)':>10} {'Metal (t)':>10}")
    print(f"   {'-' * 46}")
    total_t, total_m = 0.0, 0.0
    for cat in ["measured", "indicated", "inferred"]:
        s = statement[cat]
        print(
            f"   {cat.capitalize():<12} {s['tonnes']:>12,.0f} "
            f"{s['grade']:>10.3f} {s['metal']:>10.1f}"
        )
        total_t += s["tonnes"]
        total_m += s["metal"]
    avg_grade = total_m / total_t if total_t > 0 else 0
    print(f"   {'Total':<12} {total_t:>12,.0f} {avg_grade:>10.3f} {total_m:>10.1f}")

    # ==================================================================
    # STEP 5: Mine planning -- cut-off grade and block values
    # ==================================================================
    cu_price = 8500.0  # $/t metal
    recovery = 0.90
    mining_cost_t = 3.0  # $/t material
    processing_cost_t = 14.0  # $/t ore

    cog = breakeven_cutoff(cu_price, recovery, processing_cost_t, mining_cost_t)

    # Compute total economic value of ore blocks
    ore_value = 0.0
    ore_blocks = 0
    for i in range(bm.n_total):
        grade_frac = estimates[i] / 100.0  # Convert % to fraction
        bev = block_economic_value(
            grade=grade_frac,
            tonnage=float(block_tonnes[i]),
            price=cu_price,
            recovery=recovery,
            mining_cost=mining_cost_t,
            processing_cost=processing_cost_t,
        )
        if bev > 0:
            ore_value += bev
            ore_blocks += 1

    print("\n5. Mine Planning")
    print(f"   Break-even COG: {cog * 100:.3f}% Cu")
    print(f"   Ore blocks (positive value): {ore_blocks} of {bm.n_total}")
    print(f"   Total ore value: ${ore_value:,.0f}")

    # ==================================================================
    # STEP 6: Economic evaluation
    # ==================================================================
    capex = -20_000_000  # $20M initial CAPEX
    mine_life = 5  # 5-year mine life
    annual_revenue = ore_value / mine_life
    annual_opex = (total_t / mine_life) * (mining_cost_t + processing_cost_t)
    annual_cf = annual_revenue - annual_opex

    cashflows = [capex] + [annual_cf] * mine_life
    discount_rate = 0.10

    project_npv = npv(discount_rate, cashflows)
    project_irr = irr(cashflows)

    print("\n6. Economic Evaluation")
    print(f"   CAPEX: ${abs(capex):,.0f}")
    print(f"   Mine life: {mine_life} years")
    print(f"   Annual cash flow: ${annual_cf:,.0f}")
    print(f"   NPV (at {discount_rate * 100:.0f}%): ${project_npv:,.0f}")
    print(f"   IRR: {project_irr * 100:.1f}%")
    if project_npv > 0:
        print("   Decision: PROJECT IS VIABLE")
    else:
        print("   Decision: PROJECT IS NOT VIABLE")

    # ==================================================================
    # STEP 7: Environmental screening -- acid drainage
    # ==================================================================
    sulfur_pct = 1.8  # Total sulfur content of waste rock
    mpa = maximum_potential_acidity(sulfur_pct)
    anc = anc_func({"calcium_carbonate_pct": 3.5})
    napp_result = napp(mpa, anc)

    print("\n7. Environmental Screening (Acid Drainage)")
    print(f"   Waste rock sulfur: {sulfur_pct}%")
    print(f"   MPA: {mpa:.1f} kg H2SO4/t")
    print(f"   ANC: {anc:.1f} kg H2SO4/t")
    print(f"   NAPP: {napp_result['napp']:.1f} kg H2SO4/t")
    print(f"   Classification: {napp_result['classification']}")

    if napp_result["classification"] == "PAF":
        print("   Action: Waste requires encapsulation or treatment plan.")
    elif napp_result["classification"] == "NAF":
        print("   Action: No acid drainage mitigation required.")
    else:
        print("   Action: Additional kinetic testing recommended.")

    print(f"\n{'=' * 60}")
    print("Complete mine workflow finished.")


if __name__ == "__main__":
    main()
