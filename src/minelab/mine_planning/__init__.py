"""Pit optimization, cut-off grade, scheduling, and mine design.

This package provides tools for open-pit mine planning including pit
optimisation algorithms, cut-off grade analysis, production scheduling,
pushback design, geometric mine design, resource-to-reserve conversion,
and underground mine scheduling.
"""

from __future__ import annotations

from minelab.mine_planning.cutoff_grade import (
    breakeven_cutoff,
    lane_cutoff,
    marginal_cutoff,
)
from minelab.mine_planning.mine_design import (
    pit_geometry,
    pit_volume_tonnage,
    ramp_design,
)
from minelab.mine_planning.pit_optimization import (
    block_economic_value,
    lerchs_grossmann_2d,
    pseudoflow_3d,
)
from minelab.mine_planning.pushbacks import (
    design_pushbacks,
    nested_pit_shells,
)
from minelab.mine_planning.reserves import (
    dilution_ore_loss,
    resource_to_reserve,
)
from minelab.mine_planning.scheduling import (
    npv_schedule,
    precedence_constraints,
    schedule_by_period,
)
from minelab.mine_planning.scheduling_underground import (
    activity_on_node,
    block_value_underground,
    lateral_development_schedule,
    ore_pass_capacity,
    ramp_access_time,
    shaft_hoisting_capacity,
)
from minelab.mine_planning.underground_planning import (
    crown_pillar_thickness,
    development_advance_rate,
    long_hole_production_rate,
    mining_recovery_underground,
    stope_economic_value,
    underground_cutoff_grade,
)

__all__ = [
    # pit_optimization
    "block_economic_value",
    "lerchs_grossmann_2d",
    "pseudoflow_3d",
    # cutoff_grade
    "breakeven_cutoff",
    "lane_cutoff",
    "marginal_cutoff",
    # scheduling
    "npv_schedule",
    "precedence_constraints",
    "schedule_by_period",
    # scheduling_underground
    "activity_on_node",
    "lateral_development_schedule",
    "ore_pass_capacity",
    "shaft_hoisting_capacity",
    "ramp_access_time",
    "block_value_underground",
    # pushbacks
    "design_pushbacks",
    "nested_pit_shells",
    # mine_design
    "pit_geometry",
    "pit_volume_tonnage",
    "ramp_design",
    # underground_planning
    "crown_pillar_thickness",
    "development_advance_rate",
    "long_hole_production_rate",
    "mining_recovery_underground",
    "stope_economic_value",
    "underground_cutoff_grade",
    # reserves
    "dilution_ore_loss",
    "resource_to_reserve",
]
