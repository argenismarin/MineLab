"""Aquifer testing, pit dewatering, and groundwater chemistry for mining."""

from minelab.hydrogeology.aquifer_tests import (
    aquifer_hydraulic_conductivity,
    cooper_jacob_drawdown,
    specific_capacity,
    theis_drawdown,
    theis_recovery,
    transmissivity_from_slug,
)
from minelab.hydrogeology.groundwater_chemistry import (
    acid_mine_drainage_rate,
    dilution_attenuation_factor,
    langelier_index,
    mass_balance_water_quality,
    seepage_velocity,
)
from minelab.hydrogeology.pit_dewatering import (
    cone_of_depression_radius,
    darcy_pit_inflow,
    dewatering_power,
    dewatering_well_capacity,
    number_of_dewatering_wells,
    toth_seepage,
)

__all__ = [
    # aquifer_tests
    "theis_drawdown",
    "cooper_jacob_drawdown",
    "theis_recovery",
    "transmissivity_from_slug",
    "specific_capacity",
    "aquifer_hydraulic_conductivity",
    # pit_dewatering
    "darcy_pit_inflow",
    "toth_seepage",
    "dewatering_well_capacity",
    "number_of_dewatering_wells",
    "dewatering_power",
    "cone_of_depression_radius",
    # groundwater_chemistry
    "acid_mine_drainage_rate",
    "dilution_attenuation_factor",
    "seepage_velocity",
    "langelier_index",
    "mass_balance_water_quality",
]
