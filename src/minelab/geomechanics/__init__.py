"""Rock mass classification, slope stability, support design, and wedge analysis."""

from minelab.geomechanics.hoek_brown import (
    deformation_modulus,
    hoek_brown_intact,
    hoek_brown_parameters,
    hoek_brown_rock_mass,
    mohr_coulomb_fit,
)
from minelab.geomechanics.rock_mass_classification import (
    gsi_from_chart,
    gsi_from_rmr,
    q_system,
    rmr_bieniawski,
    smr_romana,
)
from minelab.geomechanics.slope_stability import (
    bishop_simplified,
    critical_surface_search,
    fellenius_method,
    janbu_simplified,
    pseudo_static_seismic,
    spencer_method,
)
from minelab.geomechanics.support_design import (
    pillar_strength_bieniawski,
    pillar_strength_lunder_pakalnis,
    rock_bolt_design,
    shotcrete_thickness,
    stand_up_time,
    tributary_area_stress,
)
from minelab.geomechanics.underground_excavations import (
    cable_bolt_capacity,
    failure_criterion_mohr_coulomb_ug,
    in_situ_stress_depth,
    kirsch_elastic_stress,
    plastic_zone_radius,
    shotcrete_lining_capacity,
    strength_to_stress_ratio,
    tunnel_support_pressure,
)
from minelab.geomechanics.wedge_analysis import (
    kinematic_planar,
    kinematic_toppling,
    kinematic_wedge,
    markland_test,
    stereonet_data,
    wedge_fos,
)

__all__ = [
    # hoek_brown
    "deformation_modulus",
    "hoek_brown_intact",
    "hoek_brown_parameters",
    "hoek_brown_rock_mass",
    "mohr_coulomb_fit",
    # rock_mass_classification
    "gsi_from_chart",
    "gsi_from_rmr",
    "q_system",
    "rmr_bieniawski",
    "smr_romana",
    # slope_stability
    "bishop_simplified",
    "critical_surface_search",
    "fellenius_method",
    "janbu_simplified",
    "pseudo_static_seismic",
    "spencer_method",
    # support_design
    "pillar_strength_bieniawski",
    "pillar_strength_lunder_pakalnis",
    "rock_bolt_design",
    "shotcrete_thickness",
    "stand_up_time",
    "tributary_area_stress",
    # underground_excavations
    "cable_bolt_capacity",
    "failure_criterion_mohr_coulomb_ug",
    "in_situ_stress_depth",
    "kirsch_elastic_stress",
    "plastic_zone_radius",
    "shotcrete_lining_capacity",
    "strength_to_stress_ratio",
    "tunnel_support_pressure",
    # wedge_analysis
    "kinematic_planar",
    "kinematic_toppling",
    "kinematic_wedge",
    "markland_test",
    "stereonet_data",
    "wedge_fos",
]
