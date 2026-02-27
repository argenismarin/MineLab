"""Unit conversions, mineral database, validators, grade handling, and statistics."""

from minelab.utilities.conversions import (
    angle_convert,
    density_convert,
    energy_convert,
    flowrate_convert,
    length_convert,
    mass_convert,
    pressure_convert,
    temperature_convert,
    volume_convert,
)
from minelab.utilities.grades import (
    equivalent_grade,
    gpt_to_oz_per_ton,
    gpt_to_ppm,
    grade_tonnage_curve,
    metal_content,
    oz_per_ton_to_gpt,
    percent_to_ppm,
    ppm_to_gpt,
    ppm_to_percent,
)
from minelab.utilities.mineral_db import (
    MINERAL_DB,
    get_mineral,
    get_sg,
    search_minerals,
)
from minelab.utilities.statistics import (
    capping_analysis,
    contact_analysis,
    descriptive_stats,
    log_stats,
    probability_plot,
)
from minelab.utilities.validators import (
    validate_array,
    validate_non_negative,
    validate_percentage,
    validate_positive,
    validate_probabilities,
    validate_range,
)
from minelab.utilities.visualization import (
    boxplot,
    grade_tonnage_plot,
    histogram_plot,
    scatter_plot,
    variogram_plot,
)

__all__ = [
    # conversions
    "length_convert",
    "mass_convert",
    "volume_convert",
    "pressure_convert",
    "density_convert",
    "angle_convert",
    "energy_convert",
    "flowrate_convert",
    "temperature_convert",
    # mineral_db
    "MINERAL_DB",
    "get_mineral",
    "get_sg",
    "search_minerals",
    # validators
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_percentage",
    "validate_array",
    "validate_probabilities",
    # grades
    "ppm_to_percent",
    "percent_to_ppm",
    "ppm_to_gpt",
    "gpt_to_ppm",
    "oz_per_ton_to_gpt",
    "gpt_to_oz_per_ton",
    "grade_tonnage_curve",
    "metal_content",
    "equivalent_grade",
    # statistics
    "descriptive_stats",
    "log_stats",
    "contact_analysis",
    "capping_analysis",
    "probability_plot",
    # visualization
    "histogram_plot",
    "scatter_plot",
    "variogram_plot",
    "grade_tonnage_plot",
    "boxplot",
]
