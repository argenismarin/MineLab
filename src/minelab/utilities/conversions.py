"""Unit conversion functions for mining engineering quantities.

Provides conversion between common units used in mining and geotechnical
engineering for length, mass, volume, pressure, density, angle, energy,
flow rate, and temperature.

References
----------
.. [1] Perry's Chemical Engineers' Handbook, 9th Edition, McGraw-Hill, 2019.
.. [2] SME Mining Engineering Handbook, 3rd Edition, SME, 2011.
"""

from __future__ import annotations

import numpy as np

Number = int | float | np.ndarray

# ---------------------------------------------------------------------------
# Conversion factor tables (value in *base* units per 1 of *named* unit)
# ---------------------------------------------------------------------------

_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "ft": 0.3048,
    "in": 0.0254,
    "cm": 0.01,
    "mm": 0.001,
    "yd": 0.9144,
}

_MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "lb": 0.45359237,
    "ton": 907.18474,  # short ton
    "tonne": 1000.0,  # metric tonne
    "oz": 0.028349523125,
    "g": 0.001,
}

_VOLUME_TO_M3: dict[str, float] = {
    "m3": 1.0,
    "ft3": 0.028316846592,
    "L": 0.001,
    "gal": 0.003785411784,  # US gallon
    "bbl": 0.158987294928,  # petroleum barrel
}

_PRESSURE_TO_PA: dict[str, float] = {
    "Pa": 1.0,
    "kPa": 1_000.0,
    "MPa": 1_000_000.0,
    "psi": 6_894.757293168,
    "bar": 100_000.0,
    "atm": 101_325.0,
}

_DENSITY_TO_KG_M3: dict[str, float] = {
    "kg/m3": 1.0,
    "lb/ft3": 16.01846337396,
    "g/cm3": 1_000.0,
    "t/m3": 1_000.0,
}

_ANGLE_TO_DEG: dict[str, float] = {
    "deg": 1.0,
    "rad": 180.0 / np.pi,
    "grad": 0.9,  # 1 grad = 0.9 deg
}

_ENERGY_TO_J: dict[str, float] = {
    "J": 1.0,
    "kJ": 1_000.0,
    "kWh": 3_600_000.0,
    "BTU": 1_055.05585262,
    "cal": 4.184,
}

_FLOWRATE_TO_M3S: dict[str, float] = {
    "m3/s": 1.0,
    "m3/h": 1.0 / 3600.0,
    "L/min": 1.0e-3 / 60.0,
    "gpm": 0.003785411784 / 60.0,  # US gallons per minute
    "cfm": 0.028316846592 / 60.0,  # cubic feet per minute
}


# ---------------------------------------------------------------------------
# Generic factor-based converter
# ---------------------------------------------------------------------------


def _factor_convert(
    value: Number,
    from_unit: str,
    to_unit: str,
    table: dict[str, float],
    quantity_name: str,
) -> Number:
    """Convert *value* between two units using a look-up table.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Value(s) to convert.
    from_unit : str
        Source unit key.
    to_unit : str
        Target unit key.
    table : dict
        Mapping of unit key -> factor to base unit.
    quantity_name : str
        Human-readable name used in error messages.

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).
    """
    if from_unit not in table:
        raise ValueError(f"Unknown {quantity_name} unit '{from_unit}'. Supported: {sorted(table)}")
    if to_unit not in table:
        raise ValueError(f"Unknown {quantity_name} unit '{to_unit}'. Supported: {sorted(table)}")
    if from_unit == to_unit:
        return value
    base_value = value * table[from_unit]
    return base_value / table[to_unit]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def length_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a length value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Length value(s) to convert.
    from_unit : str
        Source unit. One of ``'m'``, ``'ft'``, ``'in'``, ``'cm'``,
        ``'mm'``, ``'yd'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> length_convert(1, 'ft', 'm')
    0.3048
    >>> length_convert(1, 'yd', 'ft')
    3.0

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _LENGTH_TO_M, "length")


def mass_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a mass value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Mass value(s) to convert.
    from_unit : str
        Source unit. One of ``'kg'``, ``'lb'``, ``'ton'`` (short),
        ``'tonne'`` (metric), ``'oz'``, ``'g'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> mass_convert(1, 'tonne', 'kg')
    1000.0
    >>> round(mass_convert(1, 'lb', 'kg'), 6)
    0.453592

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _MASS_TO_KG, "mass")


def volume_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a volume value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Volume value(s) to convert.
    from_unit : str
        Source unit. One of ``'m3'``, ``'ft3'``, ``'L'``, ``'gal'``,
        ``'bbl'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> volume_convert(1, 'm3', 'L')
    1000.0

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _VOLUME_TO_M3, "volume")


def pressure_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a pressure value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Pressure value(s) to convert.
    from_unit : str
        Source unit. One of ``'Pa'``, ``'kPa'``, ``'MPa'``, ``'psi'``,
        ``'bar'``, ``'atm'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> pressure_convert(1, 'atm', 'Pa')
    101325.0
    >>> round(pressure_convert(1, 'atm', 'bar'), 5)
    1.01325

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _PRESSURE_TO_PA, "pressure")


def density_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a density value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Density value(s) to convert.
    from_unit : str
        Source unit. One of ``'kg/m3'``, ``'lb/ft3'``, ``'g/cm3'``,
        ``'t/m3'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> density_convert(1, 'g/cm3', 'kg/m3')
    1000.0

    References
    ----------
    .. [1] Perry's Chemical Engineers' Handbook, 9th ed., 2019.
    """
    return _factor_convert(value, from_unit, to_unit, _DENSITY_TO_KG_M3, "density")


def angle_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert an angle value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Angle value(s) to convert.
    from_unit : str
        Source unit. One of ``'deg'``, ``'rad'``, ``'grad'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> round(angle_convert(180, 'deg', 'rad'), 6)
    3.141593
    >>> angle_convert(100, 'grad', 'deg')
    90.0

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _ANGLE_TO_DEG, "angle")


def energy_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert an energy value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Energy value(s) to convert.
    from_unit : str
        Source unit. One of ``'J'``, ``'kJ'``, ``'kWh'``, ``'BTU'``,
        ``'cal'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> energy_convert(1, 'kWh', 'J')
    3600000.0

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    return _factor_convert(value, from_unit, to_unit, _ENERGY_TO_J, "energy")


def flowrate_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a volumetric flow-rate value between units.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Flow-rate value(s) to convert.
    from_unit : str
        Source unit. One of ``'m3/s'``, ``'m3/h'``, ``'L/min'``,
        ``'gpm'``, ``'cfm'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> flowrate_convert(1, 'm3/s', 'm3/h')
    3600.0

    References
    ----------
    .. [1] Perry's Chemical Engineers' Handbook, 9th ed., 2019.
    """
    return _factor_convert(value, from_unit, to_unit, _FLOWRATE_TO_M3S, "flowrate")


def temperature_convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a temperature value between units.

    Temperature conversions are *not* simple factor multiplications so
    they are handled as a special case with explicit formulas.

    Parameters
    ----------
    value : int, float, or numpy.ndarray
        Temperature value(s) to convert.
    from_unit : str
        Source unit. One of ``'C'``, ``'F'``, ``'K'``.
    to_unit : str
        Target unit (same options as *from_unit*).

    Returns
    -------
    int, float, or numpy.ndarray
        Converted value(s).

    Examples
    --------
    >>> temperature_convert(0, 'C', 'F')
    32.0
    >>> temperature_convert(100, 'C', 'K')
    373.15
    >>> temperature_convert(32, 'F', 'C')
    0.0

    References
    ----------
    .. [1] NIST Special Publication 811, 2008.
    """
    _valid = ("C", "F", "K")
    if from_unit not in _valid:
        raise ValueError(f"Unknown temperature unit '{from_unit}'. Supported: {_valid}")
    if to_unit not in _valid:
        raise ValueError(f"Unknown temperature unit '{to_unit}'. Supported: {_valid}")
    if from_unit == to_unit:
        return value

    # Convert to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32.0) * 5.0 / 9.0
    else:  # K
        celsius = value - 273.15

    # Convert from Celsius to target
    if to_unit == "C":
        return celsius
    elif to_unit == "F":
        return celsius * 9.0 / 5.0 + 32.0
    else:  # K
        return celsius + 273.15
