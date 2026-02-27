"""Comminution, flotation, leaching, classification, thickening, and mass balance."""

from minelab.mineral_processing.classification import (
    gates_gaudin_schuhmann,
    lynch_rao_partition,
    plitt_model,
    rosin_rammler,
    screen_efficiency,
    tromp_curve,
)
from minelab.mineral_processing.comminution import (
    ball_mill_power,
    bond_energy,
    bond_work_index,
    crusher_reduction_ratio,
    kick_energy,
    rittinger_energy,
    rod_mill_power,
    sag_mill_power,
)
from minelab.mineral_processing.flotation import (
    flotation_bank_design,
    flotation_circuit,
    flotation_first_order,
    flotation_kelsall,
    flotation_kinetics_fit,
    selectivity_index,
)
from minelab.mineral_processing.gravity_separation import (
    concentration_criterion,
    dms_cutpoint,
    humphreys_spiral_recovery,
    shaking_table_efficiency,
)
from minelab.mineral_processing.leaching import (
    acid_consumption,
    arrhenius_rate,
    cyanidation_kinetics,
    heap_leach_recovery,
    shrinking_core_diffusion,
    shrinking_core_film,
    shrinking_core_reaction,
)
from minelab.mineral_processing.magnetic_separation import (
    davis_tube_recovery,
    magnetic_susceptibility_classify,
)
from minelab.mineral_processing.mass_balance import (
    check_closure,
    multi_element_balance,
    reconcile_balance,
    three_product,
    two_product,
)
from minelab.mineral_processing.thickening import (
    coe_clevenger,
    flocculant_dosage,
    kynch_analysis,
    talmage_fitch,
)

__all__ = [
    # classification
    "gates_gaudin_schuhmann",
    "lynch_rao_partition",
    "plitt_model",
    "rosin_rammler",
    "screen_efficiency",
    "tromp_curve",
    # comminution
    "ball_mill_power",
    "bond_energy",
    "bond_work_index",
    "crusher_reduction_ratio",
    "kick_energy",
    "rittinger_energy",
    "rod_mill_power",
    "sag_mill_power",
    # flotation
    "flotation_bank_design",
    "flotation_circuit",
    "flotation_first_order",
    "flotation_kelsall",
    "flotation_kinetics_fit",
    "selectivity_index",
    # gravity separation
    "concentration_criterion",
    "dms_cutpoint",
    "humphreys_spiral_recovery",
    "shaking_table_efficiency",
    # leaching
    "acid_consumption",
    "arrhenius_rate",
    "cyanidation_kinetics",
    "heap_leach_recovery",
    "shrinking_core_diffusion",
    "shrinking_core_film",
    "shrinking_core_reaction",
    # magnetic separation
    "davis_tube_recovery",
    "magnetic_susceptibility_classify",
    # mass balance
    "check_closure",
    "multi_element_balance",
    "reconcile_balance",
    "three_product",
    "two_product",
    # thickening
    "coe_clevenger",
    "flocculant_dosage",
    "kynch_analysis",
    "talmage_fitch",
]
