"""Acid mine drainage, water balance, tailings design, dust, closure, and carbon."""

from minelab.environmental.acid_drainage import (
    acid_neutralizing_capacity,
    maximum_potential_acidity,
    nag_test_classify,
    napp,
    paste_ph_prediction,
)
from minelab.environmental.carbon import (
    blasting_emissions,
    carbon_intensity,
    diesel_emissions,
    scope1_scope2_emissions,
)
from minelab.environmental.closure import (
    acid_rock_drainage_neutralisation_cost,
    bond_amount,
    closure_cost_estimate,
    post_closure_water_management_cost,
    revegetation_success_probability,
)
from minelab.environmental.dust import (
    emission_factor_haul_roads,
    gaussian_plume,
)
from minelab.environmental.tailings import (
    tailings_beach_angle,
    tailings_storage_capacity,
)
from minelab.environmental.water_balance import (
    pit_dewatering_estimate,
    runoff_coefficient,
    site_water_balance,
)

__all__ = [
    # acid_drainage
    "maximum_potential_acidity",
    "acid_neutralizing_capacity",
    "napp",
    "nag_test_classify",
    "paste_ph_prediction",
    # water_balance
    "site_water_balance",
    "pit_dewatering_estimate",
    "runoff_coefficient",
    # tailings
    "tailings_storage_capacity",
    "tailings_beach_angle",
    # dust
    "emission_factor_haul_roads",
    "gaussian_plume",
    # closure
    "closure_cost_estimate",
    "bond_amount",
    "revegetation_success_probability",
    "acid_rock_drainage_neutralisation_cost",
    "post_closure_water_management_cost",
    # carbon
    "diesel_emissions",
    "blasting_emissions",
    "carbon_intensity",
    "scope1_scope2_emissions",
]
