"""Carbon footprint and greenhouse gas emissions for mining operations.

This module provides functions for estimating CO2 emissions from diesel
combustion, blasting agents, and electricity consumption, as well as
computing carbon intensity metrics and scope-based emissions reporting.

References
----------
.. [1] IPCC (2006). Guidelines for National Greenhouse Gas Inventories,
       Vol. 2: Energy, Ch. 3: Mobile Combustion.
.. [2] Sapag, J., Navarra, A. & Jaldín, J. (2019). "Carbon footprint of
       blasting operations in open pit mining." *Mining Engineering*,
       71(4), 40--46.
.. [3] GHG Protocol (2004). Corporate Accounting and Reporting Standard.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
)

# Default IPCC emission factor for diesel combustion (kg CO2 / litre)
_DEFAULT_DIESEL_EF: float = 2.68

# Explosive emission factors (kg CO2 per kg explosive)
_ANFO_EF: float = 0.17
_EMULSION_EF: float = 0.15


# ---------------------------------------------------------------------------
# Diesel Combustion Emissions
# ---------------------------------------------------------------------------


def diesel_emissions(
    diesel_litres: float,
    emission_factor_kgco2_per_litre: float = _DEFAULT_DIESEL_EF,
) -> dict:
    """Estimate CO2 emissions from diesel fuel combustion.

    .. math::

        \\text{CO}_2 = V \\times EF

    where *V* is the volume of diesel consumed (litres) and *EF* is the
    emission factor (kg CO2 per litre).

    Parameters
    ----------
    diesel_litres : float
        Volume of diesel consumed in litres. Must be non-negative.
    emission_factor_kgco2_per_litre : float, optional
        Emission factor in kg CO2 per litre. Default 2.68 (IPCC 2006).
        Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"co2_kg"`` : float -- Total CO2 emissions in kilograms.
        - ``"co2_tonnes"`` : float -- Total CO2 emissions in tonnes.

    Examples
    --------
    >>> result = diesel_emissions(1000)
    >>> result["co2_kg"]
    2680.0
    >>> result["co2_tonnes"]
    2.68

    References
    ----------
    .. [1] IPCC (2006). Guidelines for National Greenhouse Gas
           Inventories, Vol. 2, Ch. 3.
    """
    validate_non_negative(diesel_litres, "diesel_litres")
    validate_positive(
        emission_factor_kgco2_per_litre,
        "emission_factor_kgco2_per_litre",
    )

    co2_kg = diesel_litres * emission_factor_kgco2_per_litre
    co2_tonnes = co2_kg / 1000.0

    return {
        "co2_kg": float(co2_kg),
        "co2_tonnes": float(co2_tonnes),
    }


# ---------------------------------------------------------------------------
# Blasting Emissions
# ---------------------------------------------------------------------------


def blasting_emissions(anfo_kg: float, emulsion_kg: float) -> float:
    """Estimate CO2 emissions from explosive detonation.

    .. math::

        \\text{CO}_2 = m_{\\text{ANFO}} \\times 0.17
                     + m_{\\text{emulsion}} \\times 0.15

    Parameters
    ----------
    anfo_kg : float
        Mass of ANFO consumed in kilograms. Must be non-negative.
    emulsion_kg : float
        Mass of emulsion explosive consumed in kilograms.
        Must be non-negative.

    Returns
    -------
    float
        Total CO2 emissions in kilograms.

    Examples
    --------
    >>> blasting_emissions(1000, 500)
    245.0

    References
    ----------
    .. [1] Sapag, J. et al. (2019). "Carbon footprint of blasting
           operations in open pit mining." Mining Engineering, 71(4).
    """
    validate_non_negative(anfo_kg, "anfo_kg")
    validate_non_negative(emulsion_kg, "emulsion_kg")

    total_co2 = anfo_kg * _ANFO_EF + emulsion_kg * _EMULSION_EF
    return float(total_co2)


# ---------------------------------------------------------------------------
# Carbon Intensity
# ---------------------------------------------------------------------------


def carbon_intensity(
    total_ghg_t_co2eq: float,
    annual_production_t_metal: float,
) -> float:
    """Compute carbon intensity of metal production.

    .. math::

        CI = \\frac{\\text{GHG}_{\\text{total}}}{P_{\\text{metal}}}

    Parameters
    ----------
    total_ghg_t_co2eq : float
        Total greenhouse gas emissions in tonnes CO2-equivalent.
        Must be non-negative.
    annual_production_t_metal : float
        Annual metal production in tonnes. Must be positive.

    Returns
    -------
    float
        Carbon intensity in tonnes CO2-eq per tonne of metal produced.

    Examples
    --------
    >>> carbon_intensity(50000, 10000)
    5.0

    References
    ----------
    .. [1] GHG Protocol (2004). Corporate Accounting and Reporting
           Standard, Ch. 9.
    """
    validate_non_negative(total_ghg_t_co2eq, "total_ghg_t_co2eq")
    validate_positive(annual_production_t_metal, "annual_production_t_metal")

    return float(total_ghg_t_co2eq / annual_production_t_metal)


# ---------------------------------------------------------------------------
# Scope 1 + Scope 2 Emissions
# ---------------------------------------------------------------------------


def scope1_scope2_emissions(
    diesel_t_co2: float,
    electricity_kwh: float,
    grid_emission_factor_kg_per_kwh: float,
) -> dict:
    """Compute Scope 1 and Scope 2 greenhouse gas emissions.

    - **Scope 1** (direct): CO2 from on-site diesel combustion.
    - **Scope 2** (indirect): CO2 from purchased electricity.

    .. math::

        S_1 = \\text{diesel\\_t\\_co2}

    .. math::

        S_2 = \\frac{E \\times EF_{\\text{grid}}}{1000}

    Parameters
    ----------
    diesel_t_co2 : float
        Scope 1 emissions from diesel combustion in tonnes CO2.
        Must be non-negative.
    electricity_kwh : float
        Annual electricity consumption in kWh. Must be non-negative.
    grid_emission_factor_kg_per_kwh : float
        Grid emission factor in kg CO2 per kWh. Must be non-negative.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"scope1_tco2"`` : float -- Scope 1 emissions (t CO2).
        - ``"scope2_tco2"`` : float -- Scope 2 emissions (t CO2).
        - ``"total_tco2"`` : float -- Total Scope 1 + 2 emissions.

    Examples
    --------
    >>> result = scope1_scope2_emissions(500, 1_000_000, 0.5)
    >>> result["scope1_tco2"]
    500.0
    >>> result["scope2_tco2"]
    500.0
    >>> result["total_tco2"]
    1000.0

    References
    ----------
    .. [1] GHG Protocol (2004). Corporate Accounting and Reporting
           Standard, Ch. 4--5.
    """
    validate_non_negative(diesel_t_co2, "diesel_t_co2")
    validate_non_negative(electricity_kwh, "electricity_kwh")
    validate_non_negative(
        grid_emission_factor_kg_per_kwh,
        "grid_emission_factor_kg_per_kwh",
    )

    scope1 = diesel_t_co2
    scope2 = electricity_kwh * grid_emission_factor_kg_per_kwh / 1000.0
    total = scope1 + scope2

    return {
        "scope1_tco2": float(scope1),
        "scope2_tco2": float(scope2),
        "total_tco2": float(total),
    }
