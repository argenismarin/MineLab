"""Underground mining methods, stope design, and ground support.

This package provides tools for underground mine design including stope
stability analysis, convergence-confinement, sublevel methods, room and
pillar design, and backfill engineering.
"""

from minelab.underground_mining.backfill import (
    arching_stress,
    backfill_requirement,
    cemented_paste_strength,
    fill_pour_rate,
    hydraulic_fill_transport,
)
from minelab.underground_mining.convergence_confinement import (
    ground_reaction_curve,
    longitudinal_deformation_profile,
    rock_burst_potential,
    rock_support_interaction,
    squeezing_index,
    support_reaction_curve,
    tunnel_deformation_strain,
)
from minelab.underground_mining.room_and_pillar import (
    barrier_pillar_width,
    critical_span,
    pillar_safety_factor,
    room_and_pillar_geometry,
    subsidence_angle,
)
from minelab.underground_mining.stope_design import (
    hydraulic_radius,
    mathews_stability,
    mucking_rate,
    rill_angle,
    stope_dimensions,
    undercut_design,
)
from minelab.underground_mining.sublevel_methods import (
    block_cave_draw_rate,
    draw_ellipsoid,
    ring_blast_design,
    sublevel_interval,
    sublevel_recovery,
)

__all__ = [
    # stope_design
    "mathews_stability",
    "hydraulic_radius",
    "stope_dimensions",
    "rill_angle",
    "undercut_design",
    "mucking_rate",
    # convergence_confinement
    "ground_reaction_curve",
    "support_reaction_curve",
    "longitudinal_deformation_profile",
    "rock_support_interaction",
    "squeezing_index",
    "rock_burst_potential",
    "tunnel_deformation_strain",
    # sublevel_methods
    "sublevel_interval",
    "draw_ellipsoid",
    "sublevel_recovery",
    "ring_blast_design",
    "block_cave_draw_rate",
    # room_and_pillar
    "pillar_safety_factor",
    "room_and_pillar_geometry",
    "barrier_pillar_width",
    "critical_span",
    "subsidence_angle",
    # backfill
    "cemented_paste_strength",
    "arching_stress",
    "hydraulic_fill_transport",
    "fill_pour_rate",
    "backfill_requirement",
]
