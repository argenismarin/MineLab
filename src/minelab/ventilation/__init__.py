"""Airway resistance, network solving, fan selection, and gas dilution."""

from minelab.ventilation.airway_resistance import (
    atkinson_resistance,
    friction_factor_from_roughness,
    natural_ventilation_pressure,
    parallel_resistance,
    pressure_drop,
    series_resistance,
)
from minelab.ventilation.fan_selection import (
    fan_operating_point,
    fan_power,
    fans_in_series_parallel,
)
from minelab.ventilation.gas_dilution import (
    air_for_blasting,
    air_for_diesel,
    dust_dilution,
    methane_dilution,
)
from minelab.ventilation.network_solving import (
    hardy_cross,
    simple_network,
)
from minelab.ventilation.similarity_laws import (
    fan_affinity_laws,
    specific_speed,
)

__all__ = [
    # airway_resistance
    "atkinson_resistance",
    "pressure_drop",
    "friction_factor_from_roughness",
    "series_resistance",
    "parallel_resistance",
    "natural_ventilation_pressure",
    # network_solving
    "hardy_cross",
    "simple_network",
    # fan_selection
    "fan_operating_point",
    "fan_power",
    "fans_in_series_parallel",
    # gas_dilution
    "air_for_diesel",
    "air_for_blasting",
    "methane_dilution",
    "dust_dilution",
    # similarity_laws
    "fan_affinity_laws",
    "specific_speed",
]
