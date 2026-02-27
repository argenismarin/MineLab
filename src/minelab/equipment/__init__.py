"""Truck cycle analysis, fleet matching, productivity, fuel, conveyors, and pumps."""

from minelab.equipment.conveyors import (
    belt_conveyor_capacity,
    belt_tension,
    conveyor_power,
    conveyor_slope_limit,
    idler_spacing,
    screw_conveyor_capacity,
)
from minelab.equipment.fleet_matching import (
    match_factor,
    optimal_fleet,
)
from minelab.equipment.fuel_consumption import (
    fuel_consumption_rate,
    fuel_cost_per_tonne,
)
from minelab.equipment.productivity import (
    excavator_productivity,
    fleet_productivity,
    oee,
)
from minelab.equipment.pumps import (
    darcy_weisbach_friction,
    npsh_available,
    pump_head,
    pump_power,
    pump_specific_speed,
    slurry_pump_factor,
)
from minelab.equipment.truck_cycle import (
    rimpull_speed,
    travel_time,
    truck_cycle_time,
)

__all__ = [
    # truck_cycle
    "truck_cycle_time",
    "rimpull_speed",
    "travel_time",
    # fleet_matching
    "match_factor",
    "optimal_fleet",
    # productivity
    "fleet_productivity",
    "excavator_productivity",
    "oee",
    # fuel_consumption
    "fuel_consumption_rate",
    "fuel_cost_per_tonne",
    # conveyors
    "belt_conveyor_capacity",
    "conveyor_power",
    "belt_tension",
    "idler_spacing",
    "conveyor_slope_limit",
    "screw_conveyor_capacity",
    # pumps
    "pump_head",
    "pump_power",
    "darcy_weisbach_friction",
    "pump_specific_speed",
    "slurry_pump_factor",
    "npsh_available",
]
