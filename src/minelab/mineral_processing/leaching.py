"""Leaching kinetics and hydrometallurgical calculations.

Shrinking core model (reaction, diffusion, film control), heap leach
recovery, Arrhenius rate, cyanidation kinetics, and acid consumption.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive


def shrinking_core_reaction(
    radius: float,
    k_s: float,
    t: float,
    rho_b: float,
    c_a: float,
    b: float = 1.0,
) -> float:
    """Shrinking core model — chemical reaction control.

    Parameters
    ----------
    radius : float
        Initial particle radius (m).
    k_s : float
        Surface reaction rate constant (m/s).
    t : float
        Leaching time (s).
    rho_b : float
        Molar density of solid reactant (mol/m^3).
    c_a : float
        Bulk concentration of leaching agent (mol/m^3).
    b : float
        Stoichiometric coefficient. Default 1.

    Returns
    -------
    float
        Conversion X (fraction, 0-1).

    Examples
    --------
    >>> round(shrinking_core_reaction(0.001, 1e-4, 100, 50000, 100), 3)
    0.551

    References
    ----------
    .. [1] Levenspiel, O. (1999). Chemical Reaction Engineering. 3rd ed.,
       Ch.25.
    """
    validate_positive(radius, "radius")
    validate_positive(k_s, "k_s")
    validate_positive(rho_b, "rho_b")
    validate_positive(c_a, "c_a")

    # t = (rho_B * R) / (b * k_s * C_A) * (1 - (1-X)^(1/3))
    # Solve for X: (1-X)^(1/3) = 1 - (b * k_s * C_A * t) / (rho_B * R)
    tau = rho_b * radius / (b * k_s * c_a)
    ratio = t / tau
    ratio = min(ratio, 1.0)  # Cap at full conversion

    x = 1 - (1 - ratio) ** 3

    return float(np.clip(x, 0, 1))


def shrinking_core_diffusion(
    radius: float,
    d_e: float,
    t: float,
    rho_b: float,
    c_a: float,
    b: float = 1.0,
) -> float:
    """Shrinking core model — product layer diffusion control.

    Parameters
    ----------
    radius : float
        Initial particle radius (m).
    d_e : float
        Effective diffusivity through product layer (m^2/s).
    t : float
        Leaching time (s).
    rho_b : float
        Molar density of solid reactant (mol/m^3).
    c_a : float
        Bulk concentration of leaching agent (mol/m^3).
    b : float
        Stoichiometric coefficient. Default 1.

    Returns
    -------
    float
        Conversion X (fraction, 0-1).

    Examples
    --------
    >>> x = shrinking_core_diffusion(0.001, 1e-10, 1000, 50000, 100)
    >>> 0 < x < 1
    True

    References
    ----------
    .. [1] Levenspiel, O. (1999). Chemical Reaction Engineering. 3rd ed.,
       Ch.25.
    """
    validate_positive(radius, "radius")
    validate_positive(d_e, "d_e")
    validate_positive(rho_b, "rho_b")
    validate_positive(c_a, "c_a")

    # t = (rho_B * R^2) / (6 * b * De * CA) * (1 - 3(1-X)^(2/3) + 2(1-X))
    tau = rho_b * radius**2 / (6 * b * d_e * c_a)

    # Solve numerically: find X such that g(X) = t/tau
    # g(X) = 1 - 3(1-X)^(2/3) + 2(1-X)
    ratio = t / tau

    # Binary search for X
    x_low, x_high = 0.0, 0.9999
    for _ in range(100):
        x_mid = (x_low + x_high) / 2
        g = 1 - 3 * (1 - x_mid) ** (2 / 3) + 2 * (1 - x_mid)
        if g < ratio:
            x_low = x_mid
        else:
            x_high = x_mid

    return float(np.clip((x_low + x_high) / 2, 0, 1))


def shrinking_core_film(
    radius: float,
    k_g: float,
    t: float,
    rho_b: float,
    c_a: float,
    b: float = 1.0,
) -> float:
    """Shrinking core model — film diffusion control.

    Parameters
    ----------
    radius : float
        Initial particle radius (m).
    k_g : float
        Mass transfer coefficient (m/s).
    t : float
        Leaching time (s).
    rho_b : float
        Molar density of solid reactant (mol/m^3).
    c_a : float
        Bulk concentration of leaching agent (mol/m^3).
    b : float
        Stoichiometric coefficient. Default 1.

    Returns
    -------
    float
        Conversion X (fraction, 0-1).

    Examples
    --------
    >>> x = shrinking_core_film(0.001, 1e-3, 100, 50000, 100)
    >>> 0 <= x <= 1
    True

    References
    ----------
    .. [1] Levenspiel, O. (1999). Chemical Reaction Engineering. 3rd ed.,
       Ch.25.
    """
    validate_positive(radius, "radius")
    validate_positive(k_g, "k_g")
    validate_positive(rho_b, "rho_b")
    validate_positive(c_a, "c_a")

    # t = (rho_B * R) / (3 * b * kg * CA) * X
    tau = rho_b * radius / (3 * b * k_g * c_a)
    x = t / tau

    return float(np.clip(x, 0, 1))


def heap_leach_recovery(
    column_recoveries: np.ndarray,
    column_times: np.ndarray,
    target_time: float,
) -> float:
    """Extrapolate heap leach recovery from column test data.

    Parameters
    ----------
    column_recoveries : np.ndarray
        Recovery values from column test (fractions, 0-1).
    column_times : np.ndarray
        Corresponding times (days).
    target_time : float
        Target heap leach time (days).

    Returns
    -------
    float
        Estimated recovery at target time.

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0, 30, 60, 90, 120])
    >>> rec = np.array([0, 0.3, 0.5, 0.6, 0.65])
    >>> r = heap_leach_recovery(rec, times, 180)
    >>> 0 < r < 1
    True

    References
    ----------
    .. [1] Ghorbani, Y. et al. (2016). Large particle effects in
       chemical/biochemical heap leach processes. Minerals Engineering.
    """
    column_recoveries = np.asarray(column_recoveries, dtype=float)
    column_times = np.asarray(column_times, dtype=float)

    # Linear interpolation/extrapolation
    return float(np.clip(np.interp(target_time, column_times, column_recoveries), 0, 1))


def arrhenius_rate(
    a: float,
    ea: float,
    t: float,
) -> float:
    """Arrhenius rate constant.

    Parameters
    ----------
    a : float
        Pre-exponential factor (same units as k).
    ea : float
        Activation energy (J/mol).
    t : float
        Temperature (K).

    Returns
    -------
    float
        Rate constant k.

    Examples
    --------
    >>> k = arrhenius_rate(1e10, 50000, 298)
    >>> k > 0
    True

    References
    ----------
    .. [1] Levenspiel, O. (1999). Chemical Reaction Engineering. 3rd ed.
    """
    validate_positive(a, "a")
    validate_positive(ea, "ea")
    validate_positive(t, "t")

    r_gas = 8.314  # J/(mol*K)
    # k = A * exp(-Ea / (R*T))
    return float(a * np.exp(-ea / (r_gas * t)))


def cyanidation_kinetics(
    grade: float,
    cn_conc: float,
    time: float,
    k: float,
) -> float:
    """First-order cyanidation leach kinetics.

    Parameters
    ----------
    grade : float
        Head grade (g/t Au or Ag).
    cn_conc : float
        Cyanide concentration (g/L).
    time : float
        Leach time (hours).
    k : float
        Rate constant (1/h).

    Returns
    -------
    float
        Recovery (fraction, 0-1).

    Examples
    --------
    >>> round(cyanidation_kinetics(5.0, 0.5, 24, 0.1), 2)
    0.91

    References
    ----------
    .. [1] Habashi, F. (1987). "One hundred years of cyanidation."
       CIM Bulletin, 80(905), 108-114.
    """
    validate_positive(grade, "grade")
    validate_positive(cn_conc, "cn_conc")
    validate_positive(k, "k")

    # First-order: R = 1 - exp(-k*t)
    return float(1 - np.exp(-k * time))


def acid_consumption(
    ite_percent: float,
    ite_acid_factor: float = 30.6,
) -> float:
    """Estimated acid consumption from sulfide mineralogy.

    Parameters
    ----------
    calcite_percent : float
        Percent calcite or acid-consuming gangue.
    acid_factor : float
        Acid consumption factor (kg H2SO4 per % calcite per tonne).
        Default 30.6 from AMIRA P387A.

    Returns
    -------
    float
        Acid consumption (kg H2SO4/t).

    Examples
    --------
    >>> acid_consumption(2.0)
    61.2

    References
    ----------
    .. [1] Jergensen (1999). Copper leaching, solvent extraction,
       and electrowinning technology.
    """
    return float(ite_percent * ite_acid_factor)
