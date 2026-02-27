"""Mine closure cost estimation and rehabilitation planning.

This module provides functions for estimating mine closure costs, financial
assurance (bonding), revegetation success probability, acid rock drainage
neutralisation costs, and post-closure water management costs.

References
----------
.. [1] INAP (2014). Global Acid Rock Drainage Guide (GARD Guide).
.. [2] ICMM (2019). Integrated Mine Closure: Good Practice Guide, 2nd ed.
.. [3] Tongway, D.J. & Ludwig, J.A. (2011). Restoring Disturbed Landscapes.
       Island Press.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# Disturbance type multipliers for closure cost estimation
_DISTURBANCE_MULTIPLIERS: dict[str, float] = {
    "tailings": 1.5,
    "waste_dump": 1.2,
    "pit": 2.0,
    "infrastructure": 0.8,
}


# ---------------------------------------------------------------------------
# Closure Cost Estimate
# ---------------------------------------------------------------------------


def closure_cost_estimate(
    disturbed_area_ha: float,
    disturbance_type: str,
    unit_cost: float,
) -> float:
    """Estimate mine closure cost based on disturbed area and type.

    Applies a disturbance-type multiplier to the base cost:

    .. math::

        C = A \\times C_u \\times m

    where *A* is the disturbed area (ha), *C_u* is the unit cost per
    hectare, and *m* is the multiplier for the disturbance type.

    Parameters
    ----------
    disturbed_area_ha : float
        Total disturbed area in hectares. Must be positive.
    disturbance_type : str
        Type of disturbance. Recognised types and their multipliers:
        ``"tailings"`` (1.5), ``"waste_dump"`` (1.2), ``"pit"`` (2.0),
        ``"infrastructure"`` (0.8). Any other string defaults to 1.0.
    unit_cost : float
        Base rehabilitation cost per hectare in currency units.
        Must be positive.

    Returns
    -------
    float
        Estimated total closure cost in currency units.

    Examples
    --------
    >>> closure_cost_estimate(100, "tailings", 5000)
    750000.0

    >>> closure_cost_estimate(50, "pit", 8000)
    800000.0

    References
    ----------
    .. [1] INAP (2014). Global Acid Rock Drainage Guide (GARD Guide).
    .. [2] ICMM (2019). Integrated Mine Closure: Good Practice Guide.
    """
    validate_positive(disturbed_area_ha, "disturbed_area_ha")
    validate_positive(unit_cost, "unit_cost")

    multiplier = _DISTURBANCE_MULTIPLIERS.get(disturbance_type.lower(), 1.0)
    return float(disturbed_area_ha * unit_cost * multiplier)


# ---------------------------------------------------------------------------
# Bond Amount (Present Value of Closure Cost)
# ---------------------------------------------------------------------------


def bond_amount(
    npv_closure_cost: float,
    discount_rate: float,
    years_to_closure: float,
) -> float:
    """Compute the present-value closure bond amount.

    .. math::

        B = \\frac{C_{\\text{closure}}}{(1 + r)^t}

    where *C_closure* is the estimated closure cost, *r* is the
    discount rate, and *t* is the number of years to closure.

    Parameters
    ----------
    npv_closure_cost : float
        Estimated total closure cost in future currency units.
        Must be positive.
    discount_rate : float
        Annual discount rate as a decimal (e.g. 0.08 for 8 %).
        Must be positive.
    years_to_closure : float
        Years remaining until mine closure. Must be positive.

    Returns
    -------
    float
        Required bond amount in present-value currency units.

    Examples
    --------
    >>> round(bond_amount(1_000_000, 0.08, 10), 2)
    463193.49

    References
    ----------
    .. [1] ICMM (2019). Integrated Mine Closure: Good Practice Guide.
    """
    validate_positive(npv_closure_cost, "npv_closure_cost")
    validate_positive(discount_rate, "discount_rate")
    validate_positive(years_to_closure, "years_to_closure")

    return float(npv_closure_cost / (1 + discount_rate) ** years_to_closure)


# ---------------------------------------------------------------------------
# Revegetation Success Probability
# ---------------------------------------------------------------------------


def revegetation_success_probability(
    rainfall_mm: float,
    slope_pct: float,
    topsoil_depth_mm: float,
    seed_mix_quality: float,
) -> dict:
    """Estimate revegetation success probability using empirical factors.

    Four independent factors are combined multiplicatively:

    .. math::

        P = f_{\\text{rain}} \\times f_{\\text{slope}} \\times
            f_{\\text{soil}} \\times q

    where each factor is clipped to [0, 1]:

    - Rain factor: ``min(1.0, rainfall_mm / 600)``
    - Slope factor: ``max(0.3, 1 - slope_pct / 50)``
    - Soil factor: ``min(1.0, topsoil_depth_mm / 300)``
    - Quality factor: ``seed_mix_quality`` (already in [0, 1])

    Parameters
    ----------
    rainfall_mm : float
        Mean annual rainfall in mm. Must be non-negative.
    slope_pct : float
        Average slope as a percentage. Must be non-negative.
    topsoil_depth_mm : float
        Applied topsoil depth in mm. Must be non-negative.
    seed_mix_quality : float
        Seed mix quality index in [0, 1]. Must be in [0, 1].

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"probability"`` : float -- Overall success probability [0, 1].
        - ``"rain_factor"`` : float -- Rainfall contribution factor.
        - ``"slope_factor"`` : float -- Slope contribution factor.
        - ``"soil_factor"`` : float -- Topsoil depth contribution factor.
        - ``"risk_class"`` : str -- ``"low"`` (P >= 0.7),
          ``"medium"`` (0.4 <= P < 0.7), or ``"high"`` (P < 0.4).

    Examples
    --------
    >>> result = revegetation_success_probability(600, 10, 300, 1.0)
    >>> result["probability"]
    0.8
    >>> result["risk_class"]
    'low'

    References
    ----------
    .. [1] Tongway, D.J. & Ludwig, J.A. (2011). Restoring Disturbed
           Landscapes. Island Press.
    .. [2] ICMM (2019). Integrated Mine Closure: Good Practice Guide.
    """
    validate_non_negative(rainfall_mm, "rainfall_mm")
    validate_non_negative(slope_pct, "slope_pct")
    validate_non_negative(topsoil_depth_mm, "topsoil_depth_mm")
    validate_range(seed_mix_quality, 0, 1, "seed_mix_quality")

    rain_factor = min(1.0, rainfall_mm / 600.0)
    slope_factor = max(0.3, 1.0 - slope_pct / 50.0)
    soil_factor = min(1.0, topsoil_depth_mm / 300.0)

    probability = rain_factor * slope_factor * soil_factor * seed_mix_quality

    if probability >= 0.7:
        risk_class = "low"
    elif probability >= 0.4:
        risk_class = "medium"
    else:
        risk_class = "high"

    return {
        "probability": float(probability),
        "rain_factor": float(rain_factor),
        "slope_factor": float(slope_factor),
        "soil_factor": float(soil_factor),
        "risk_class": risk_class,
    }


# ---------------------------------------------------------------------------
# Acid Rock Drainage Neutralisation Cost
# ---------------------------------------------------------------------------


def acid_rock_drainage_neutralisation_cost(
    napp_kg_t: float,
    tonnes_acid_forming: float,
    lime_cost_per_tonne: float,
) -> dict:
    """Estimate lime neutralisation cost from NAPP values.

    Calculates the mass of lime (CaCO3) required to neutralise
    acid-producing waste material and the associated cost:

    .. math::

        R = \\frac{\\text{NAPP}}{1000} \\times 1.02

    .. math::

        M_{\\text{lime}} = R \\times T_{\\text{waste}}

    where *R* is the lime ratio (t CaCO3 per t waste), 1.02 is the
    stoichiometric factor, and *T_waste* is the total acid-forming
    tonnage.

    Parameters
    ----------
    napp_kg_t : float
        Net Acid Producing Potential in kg H2SO4 per tonne. Must be
        positive (material must be acid-forming).
    tonnes_acid_forming : float
        Total tonnes of acid-forming material. Must be positive.
    lime_cost_per_tonne : float
        Cost of lime per tonne in currency units. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"lime_required_tonnes"`` : float -- Mass of CaCO3 needed.
        - ``"total_cost"`` : float -- Total lime procurement cost.
        - ``"cost_per_tonne_waste"`` : float -- Cost per tonne of waste.

    Examples
    --------
    >>> result = acid_rock_drainage_neutralisation_cost(30, 1_000_000, 50)
    >>> round(result["lime_required_tonnes"], 0)
    30600.0
    >>> round(result["cost_per_tonne_waste"], 4)
    1.53

    References
    ----------
    .. [1] INAP (2014). Global Acid Rock Drainage Guide (GARD Guide).
    .. [2] AMIRA P387A (2002). ARD Test Handbook.
    """
    validate_positive(napp_kg_t, "napp_kg_t")
    validate_positive(tonnes_acid_forming, "tonnes_acid_forming")
    validate_positive(lime_cost_per_tonne, "lime_cost_per_tonne")

    lime_ratio = napp_kg_t / 1000.0 * 1.02
    lime_tonnes = lime_ratio * tonnes_acid_forming
    total_cost = lime_tonnes * lime_cost_per_tonne
    cost_per_tonne_waste = total_cost / tonnes_acid_forming

    return {
        "lime_required_tonnes": float(lime_tonnes),
        "total_cost": float(total_cost),
        "cost_per_tonne_waste": float(cost_per_tonne_waste),
    }


# ---------------------------------------------------------------------------
# Post-Closure Water Management Cost
# ---------------------------------------------------------------------------


def post_closure_water_management_cost(
    seepage_rate_m3y: float,
    treatment_cost_per_m3: float,
    years: float,
) -> float:
    """Estimate total post-closure water treatment cost.

    .. math::

        C = Q \\times C_t \\times T

    where *Q* is the annual seepage rate (m3/y), *C_t* is the unit
    treatment cost (per m3), and *T* is the treatment duration (years).

    Parameters
    ----------
    seepage_rate_m3y : float
        Annual seepage rate in cubic metres per year. Must be positive.
    treatment_cost_per_m3 : float
        Water treatment cost per cubic metre. Must be positive.
    years : float
        Duration of post-closure treatment in years. Must be positive.

    Returns
    -------
    float
        Total post-closure water management cost in currency units.

    Examples
    --------
    >>> post_closure_water_management_cost(10000, 2.5, 20)
    500000.0

    References
    ----------
    .. [1] INAP (2014). Global Acid Rock Drainage Guide (GARD Guide),
           Ch. 7: Prediction.
    """
    validate_positive(seepage_rate_m3y, "seepage_rate_m3y")
    validate_positive(treatment_cost_per_m3, "treatment_cost_per_m3")
    validate_positive(years, "years")

    return float(seepage_rate_m3y * treatment_cost_per_m3 * years)
