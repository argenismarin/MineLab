"""Example 07 -- Open pit optimization with a synthetic block model.

This script demonstrates the pit optimization and scheduling workflow:
  1. Create a synthetic 2D block model with grades.
  2. Compute economic block values (revenue minus costs).
  3. Calculate the break-even cut-off grade.
  4. Run the Lerchs-Grossmann 2D algorithm for the ultimate pit limit.
  5. Schedule the pit into mining periods.
  6. Compute the NPV of the scheduled cash flows.

Functions used from ``minelab.mine_planning`` and ``minelab.economics``.
"""

from __future__ import annotations

import numpy as np

from minelab.economics import npv
from minelab.mine_planning import (
    block_economic_value,
    breakeven_cutoff,
    lerchs_grossmann_2d,
    npv_schedule,
    schedule_by_period,
)


def main() -> None:
    print("=" * 60)
    print("OPEN PIT OPTIMIZATION -- SYNTHETIC BLOCK MODEL")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create a synthetic 2D block model
    # ------------------------------------------------------------------
    # Block model dimensions: 8 benches (rows) x 12 columns
    n_levels = 8
    n_cols = 12
    np.random.seed(42)

    # Simulate an orebody: high grades in the center-bottom, waste on sides
    grade_model = np.zeros((n_levels, n_cols))
    for i in range(n_levels):
        for j in range(n_cols):
            # Distance from the orebody center (row 5, col 6)
            dist = np.sqrt((i - 5) ** 2 + (j - 6) ** 2)
            # Grade decreases with distance, add some noise
            base_grade = max(0, 0.8 - 0.12 * dist)
            noise = np.random.normal(0, 0.03)
            grade_model[i, j] = max(0, base_grade + noise)

    print("\n1. Synthetic Block Model")
    print(f"   Dimensions: {n_levels} levels x {n_cols} columns")
    print(f"   Grade range: {grade_model.min():.4f} to {grade_model.max():.4f} (% Cu)")
    print(f"   Mean grade (all blocks): {grade_model.mean():.4f} % Cu")

    # ------------------------------------------------------------------
    # 2. Economic parameters and cut-off grade
    # ------------------------------------------------------------------
    cu_price = 8000.0  # Copper price ($/tonne of metal)
    recovery = 0.88  # Metallurgical recovery (88%)
    mining_cost = 2.50  # Mining cost ($/tonne of material)
    processing_cost = 12.0  # Processing cost ($/tonne of ore)
    ga_cost = 1.50  # General & admin cost ($/tonne)
    tonnage_per_block = 10000.0  # Each block = 10,000 tonnes

    cog = breakeven_cutoff(
        price=cu_price,
        recovery=recovery,
        processing_cost=processing_cost,
        mining_cost=mining_cost,
        ga_cost=ga_cost,
    )
    print("\n2. Economic Parameters")
    print(f"   Cu price:        ${cu_price:,.0f}/t metal")
    print(f"   Recovery:        {recovery * 100:.0f}%")
    print(f"   Mining cost:     ${mining_cost:.2f}/t material")
    print(f"   Processing cost: ${processing_cost:.2f}/t ore")
    print(f"   Break-even COG:  {cog * 100:.3f}% Cu ({cog:.5f} fraction)")

    # ------------------------------------------------------------------
    # 3. Compute block economic values
    # ------------------------------------------------------------------
    # For each block, compute its economic value
    block_values = np.zeros((n_levels, n_cols))
    for i in range(n_levels):
        for j in range(n_cols):
            grade = grade_model[i, j] / 100.0  # Convert % to fraction
            block_values[i, j] = block_economic_value(
                grade=grade,
                tonnage=tonnage_per_block,
                price=cu_price,
                recovery=recovery,
                mining_cost=mining_cost,
                processing_cost=processing_cost,
            )

    n_positive = np.sum(block_values > 0)
    n_negative = np.sum(block_values <= 0)
    print("\n3. Block Economic Values")
    print(f"   Positive-value blocks (ore): {n_positive}")
    print(f"   Negative-value blocks (waste): {n_negative}")
    print(f"   Total undiscounted value (all): ${block_values.sum():,.0f}")
    print(f"   Max block value: ${block_values.max():,.0f}")
    print(f"   Min block value: ${block_values.min():,.0f}")

    # ------------------------------------------------------------------
    # 4. Lerchs-Grossmann 2D pit optimization
    # ------------------------------------------------------------------
    slope_angles = (50.0, 50.0)  # 50-degree overall slopes (left, right)
    lg_result = lerchs_grossmann_2d(block_values, slope_angles)

    pit_mask = lg_result["pit_mask"]
    total_pit_value = lg_result["total_value"]
    n_pit_blocks = pit_mask.sum()

    # Compute ore and waste within the pit
    ore_mask = pit_mask & (block_values > 0)
    waste_mask = pit_mask & (block_values <= 0)
    ore_tonnes = ore_mask.sum() * tonnage_per_block
    waste_tonnes = waste_mask.sum() * tonnage_per_block
    strip_ratio = waste_tonnes / ore_tonnes if ore_tonnes > 0 else 0

    print("\n4. Lerchs-Grossmann 2D Optimal Pit")
    print(f"   Slope angles: {slope_angles[0]}deg (L), {slope_angles[1]}deg (R)")
    print(f"   Blocks in pit: {n_pit_blocks} of {n_levels * n_cols}")
    print(f"   Ore blocks:   {ore_mask.sum()}")
    print(f"   Waste blocks: {waste_mask.sum()}")
    print(f"   Ore tonnage:   {ore_tonnes:,.0f} t")
    print(f"   Waste tonnage: {waste_tonnes:,.0f} t")
    print(f"   Strip ratio:  {strip_ratio:.2f}:1 (waste:ore)")
    print(f"   Total pit value (undiscounted): ${total_pit_value:,.0f}")

    # Print a visual representation of the pit
    print("\n   Pit cross-section (O=ore in pit, W=waste in pit, .=outside):")
    for i in range(n_levels):
        row_str = "   "
        for j in range(n_cols):
            if ore_mask[i, j]:
                row_str += "O "
            elif waste_mask[i, j]:
                row_str += "W "
            else:
                row_str += ". "
        print(row_str)

    # ------------------------------------------------------------------
    # 5. Production scheduling
    # ------------------------------------------------------------------
    n_periods = 4
    # Each period can handle roughly 1/4 of the pit value capacity
    total_abs_value = np.sum(np.abs(block_values[pit_mask]))
    cap_per_period = total_abs_value / n_periods * 1.1  # 10% buffer
    capacities = [cap_per_period] * n_periods

    sched_result = schedule_by_period(block_values, pit_mask, capacities, n_periods)
    period_values = sched_result["period_values"]

    print(f"\n5. Production Schedule ({n_periods} periods)")
    for p, val in enumerate(period_values, 1):
        print(f"   Period {p}: ${val:>12,.0f}")
    print(f"   Total scheduled: ${sum(period_values):,.0f}")

    # ------------------------------------------------------------------
    # 6. NPV calculation
    # ------------------------------------------------------------------
    discount_rate = 0.10  # 10% discount rate

    # NPV using the mine_planning scheduling NPV function
    schedule_npv = npv_schedule(period_values, discount_rate)

    # Also calculate using the economics.npv function (includes time-0 CAPEX)
    capex = -15_000_000  # Initial capital investment
    full_cashflows = [capex] + period_values
    project_npv = npv(discount_rate, full_cashflows)

    print("\n6. Economic Evaluation")
    print(f"   Discount rate: {discount_rate * 100:.0f}%")
    print(f"   NPV of scheduled pit (no CAPEX): ${schedule_npv:,.0f}")
    print(f"   CAPEX: ${capex:,.0f}")
    print(f"   Project NPV (with CAPEX): ${project_npv:,.0f}")

    if project_npv > 0:
        print("   Decision: PROJECT IS ECONOMICALLY VIABLE")
    else:
        print("   Decision: PROJECT IS NOT VIABLE at current prices")

    print(f"\n{'=' * 60}")
    print("Pit optimization complete.")


if __name__ == "__main__":
    main()
