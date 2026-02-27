"""Groundwater chemistry functions for mining applications.

This module provides calculations related to acid mine drainage prediction,
contaminant transport, and water quality mixing commonly encountered in
mining hydrogeology.

References
----------
.. [1] Nicholson, R.V. et al. (1988). An analysis of the factors affecting
       the rate of pyrite oxidation. *Preprints, Div. Fuel Chemistry, ACS*,
       33(4), 202--207.
.. [2] Langelier, W.F. (1936). The analytical control of anti-corrosion
       water treatment. *J. AWWA*, 28(10), 1500--1521.
.. [3] Freeze, R.A. & Cherry, J.A. (1979). *Groundwater*. Prentice-Hall.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import (
    validate_array,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Acid Mine Drainage Rate
# ---------------------------------------------------------------------------


def acid_mine_drainage_rate(
    pyrite_fraction: float,
    surface_area: float,
    oxidation_rate: float,
    density: float,
) -> float:
    """Estimate acid generation rate from pyrite oxidation.

    AMD_rate = pyrite_fraction * surface_area * oxidation_rate * density

    Parameters
    ----------
    pyrite_fraction : float
        Fraction of pyrite in the rock mass, in [0, 1].
    surface_area : float
        Specific surface area in m2/kg.  Must be > 0.
    oxidation_rate : float
        Oxidation rate in kg/(m2*day).  Must be > 0.
    density : float
        Rock bulk density in kg/m3.  Must be > 0.

    Returns
    -------
    float
        Acid generation rate in kg/day per m3 of rock.

    References
    ----------
    .. [1] Nicholson, R.V. et al. (1988). An analysis of the factors
           affecting the rate of pyrite oxidation. *Preprints, Div. Fuel
           Chemistry, ACS*, 33(4), 202--207.
    """
    validate_range(pyrite_fraction, 0.0, 1.0, "pyrite_fraction")
    validate_positive(surface_area, "surface_area")
    validate_positive(oxidation_rate, "oxidation_rate")
    validate_positive(density, "density")

    rate = pyrite_fraction * surface_area * oxidation_rate * density
    return float(rate)


# ---------------------------------------------------------------------------
# Dilution Attenuation Factor
# ---------------------------------------------------------------------------


def dilution_attenuation_factor(
    C_source: float,  # noqa: N803
    K_d: float,  # noqa: N803
    rho_bulk: float,
    theta: float,
) -> float:
    """Compute the retardation-based dilution attenuation factor.

    R = 1 + rho_bulk * K_d / theta

    The DAF equals the retardation factor R, representing how many
    times a contaminant concentration is reduced relative to the
    source by sorption-induced retardation.

    Parameters
    ----------
    C_source : float
        Source concentration in mg/L.  Must be > 0.
    K_d : float
        Distribution coefficient in L/kg.  Must be >= 0.
    rho_bulk : float
        Bulk density in kg/L.  Must be > 0.
    theta : float
        Effective porosity (dimensionless), in (0, 1].

    Returns
    -------
    float
        Dilution attenuation factor (dimensionless, >= 1).

    References
    ----------
    .. [1] Freeze, R.A. & Cherry, J.A. (1979). *Groundwater*.
           Prentice-Hall. Ch. 9.
    """
    validate_positive(C_source, "C_source")
    if K_d < 0:
        raise ValueError(f"'K_d' must be non-negative, got {K_d}.")
    validate_positive(rho_bulk, "rho_bulk")
    validate_range(theta, 0.01, 1.0, "theta")

    retardation = 1.0 + rho_bulk * K_d / theta
    return float(retardation)


# ---------------------------------------------------------------------------
# Seepage Velocity
# ---------------------------------------------------------------------------


def seepage_velocity(
    K: float,  # noqa: N803
    gradient: float,
    porosity: float,
) -> float:
    """Compute seepage (pore water) velocity.

    v = K * gradient / porosity

    This is the actual velocity of water through pore spaces, as
    opposed to the Darcy velocity (K * i).

    Parameters
    ----------
    K : float
        Hydraulic conductivity in m/day.  Must be > 0.
    gradient : float
        Hydraulic gradient (dimensionless).  Must be > 0.
    porosity : float
        Effective porosity (dimensionless), in (0, 1].

    Returns
    -------
    float
        Seepage velocity in m/day.

    References
    ----------
    .. [1] Freeze, R.A. & Cherry, J.A. (1979). *Groundwater*.
           Prentice-Hall. Ch. 2.
    """
    validate_positive(K, "K")
    validate_positive(gradient, "gradient")
    validate_range(porosity, 0.01, 1.0, "porosity")

    return float(K * gradient / porosity)


# ---------------------------------------------------------------------------
# Langelier Saturation Index
# ---------------------------------------------------------------------------


def langelier_index(
    pH: float,  # noqa: N803
    temp_c: float,
    ca_ppm: float,
    total_alk_ppm: float,
    tds_ppm: float,
) -> float:
    """Compute the Langelier Saturation Index (LSI).

    LSI = pH - pHs

    where:
        pHs = (9.3 + A + B) - (C + D)
        A = (log10(TDS) - 1) / 10
        B = -13.12 * log10(temp_C + 273) + 34.55
        C = log10(Ca as CaCO3) - 0.4
        D = log10(alkalinity as CaCO3)

    Positive LSI indicates scale-forming tendency; negative indicates
    corrosive water.

    Parameters
    ----------
    pH : float
        Measured pH of the water, in [0, 14].
    temp_c : float
        Water temperature in degrees Celsius.  Must be > 0.
    ca_ppm : float
        Calcium concentration as CaCO3 in mg/L (ppm).  Must be > 0.
    total_alk_ppm : float
        Total alkalinity as CaCO3 in mg/L (ppm).  Must be > 0.
    tds_ppm : float
        Total dissolved solids in mg/L (ppm).  Must be > 0.

    Returns
    -------
    float
        Langelier Saturation Index (dimensionless).

    References
    ----------
    .. [1] Langelier, W.F. (1936). The analytical control of
           anti-corrosion water treatment. *J. AWWA*, 28(10),
           1500--1521.
    """
    validate_range(pH, 0.0, 14.0, "pH")
    validate_positive(temp_c, "temp_c")
    validate_positive(ca_ppm, "ca_ppm")
    validate_positive(total_alk_ppm, "total_alk_ppm")
    validate_positive(tds_ppm, "tds_ppm")

    a = (math.log10(tds_ppm) - 1.0) / 10.0
    b = -13.12 * math.log10(temp_c + 273.0) + 34.55
    c = math.log10(ca_ppm) - 0.4
    d = math.log10(total_alk_ppm)

    ph_s = (9.3 + a + b) - (c + d)
    lsi = pH - ph_s
    return float(lsi)


# ---------------------------------------------------------------------------
# Mass Balance Water Quality
# ---------------------------------------------------------------------------


def mass_balance_water_quality(
    flows: list,
    concentrations: list,
) -> dict:
    """Flow-weighted mixing calculation for multiple water streams.

    C_mix = sum(Q_i * C_i) / sum(Q_i)
    Q_total = sum(Q_i)

    Parameters
    ----------
    flows : list
        Flow rates in m3/day for each stream.  All must be > 0.
        Must have at least 1 element.
    concentrations : list
        Contaminant concentrations in mg/L for each stream.
        Must be >= 0 and same length as *flows*.

    Returns
    -------
    dict
        ``total_flow`` : float
            Sum of all flow rates in m3/day.
        ``mixed_concentration`` : float
            Flow-weighted average concentration in mg/L.

    Raises
    ------
    ValueError
        If arrays are empty or have mismatched lengths.

    References
    ----------
    .. [1] Freeze, R.A. & Cherry, J.A. (1979). *Groundwater*.
           Prentice-Hall. Ch. 9.
    """
    flows_arr = validate_array(flows, "flows", min_length=1)
    conc_arr = validate_array(concentrations, "concentrations", min_length=1)

    if len(flows_arr) != len(conc_arr):
        raise ValueError(
            "'flows' and 'concentrations' must have the same length, "
            f"got {len(flows_arr)} and {len(conc_arr)}."
        )

    for i, q in enumerate(flows_arr):
        if q <= 0:
            raise ValueError(f"All flows must be positive, got flows[{i}]={q}.")
    for i, c in enumerate(conc_arr):
        if c < 0:
            raise ValueError(
                f"All concentrations must be non-negative, got concentrations[{i}]={c}."
            )

    total_flow = float(flows_arr.sum())
    mixed_conc = float((flows_arr * conc_arr).sum() / total_flow)
    return {
        "total_flow": total_flow,
        "mixed_concentration": mixed_conc,
    }
