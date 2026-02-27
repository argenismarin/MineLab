"""Mineral properties database for mining engineering applications.

Provides a look-up table of common minerals with name, chemical formula,
specific gravity (SG), Mohs hardness, and crystal system.

References
----------
.. [1] Dana's New Mineralogy, 8th Edition, Wiley, 1997.
.. [2] Klein, C. & Dutrow, B., Manual of Mineral Science, 23rd ed., 2007.
.. [3] Mindat.org — mineral database (accessed 2024).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Database of 55 common minerals
# ---------------------------------------------------------------------------

MINERAL_DB: dict[str, dict[str, Any]] = {
    "quartz": {
        "name": "Quartz",
        "formula": "SiO2",
        "sg": 2.65,
        "hardness": 7.0,
        "crystal_system": "hexagonal",
    },
    "feldspar": {
        "name": "Feldspar",
        "formula": "KAlSi3O8",
        "sg": 2.56,
        "hardness": 6.0,
        "crystal_system": "monoclinic",
    },
    "orthoclase": {
        "name": "Orthoclase",
        "formula": "KAlSi3O8",
        "sg": 2.56,
        "hardness": 6.0,
        "crystal_system": "monoclinic",
    },
    "plagioclase": {
        "name": "Plagioclase",
        "formula": "NaAlSi3O8-CaAl2Si2O8",
        "sg": 2.68,
        "hardness": 6.0,
        "crystal_system": "triclinic",
    },
    "muscovite": {
        "name": "Muscovite",
        "formula": "KAl2(AlSi3O10)(OH)2",
        "sg": 2.82,
        "hardness": 2.5,
        "crystal_system": "monoclinic",
    },
    "biotite": {
        "name": "Biotite",
        "formula": "K(Mg,Fe)3(AlSi3O10)(OH)2",
        "sg": 3.0,
        "hardness": 2.5,
        "crystal_system": "monoclinic",
    },
    "hornblende": {
        "name": "Hornblende",
        "formula": "Ca2(Mg,Fe,Al)5(Al,Si)8O22(OH)2",
        "sg": 3.2,
        "hardness": 5.5,
        "crystal_system": "monoclinic",
    },
    "augite": {
        "name": "Augite",
        "formula": "(Ca,Na)(Mg,Fe,Al)(Si,Al)2O6",
        "sg": 3.3,
        "hardness": 5.5,
        "crystal_system": "monoclinic",
    },
    "olivine": {
        "name": "Olivine",
        "formula": "(Mg,Fe)2SiO4",
        "sg": 3.3,
        "hardness": 6.5,
        "crystal_system": "orthorhombic",
    },
    "calcite": {
        "name": "Calcite",
        "formula": "CaCO3",
        "sg": 2.71,
        "hardness": 3.0,
        "crystal_system": "hexagonal",
    },
    "dolomite": {
        "name": "Dolomite",
        "formula": "CaMg(CO3)2",
        "sg": 2.85,
        "hardness": 3.5,
        "crystal_system": "hexagonal",
    },
    "gypsum": {
        "name": "Gypsum",
        "formula": "CaSO4·2H2O",
        "sg": 2.32,
        "hardness": 2.0,
        "crystal_system": "monoclinic",
    },
    "anhydrite": {
        "name": "Anhydrite",
        "formula": "CaSO4",
        "sg": 2.98,
        "hardness": 3.5,
        "crystal_system": "orthorhombic",
    },
    "halite": {
        "name": "Halite",
        "formula": "NaCl",
        "sg": 2.16,
        "hardness": 2.5,
        "crystal_system": "cubic",
    },
    "fluorite": {
        "name": "Fluorite",
        "formula": "CaF2",
        "sg": 3.18,
        "hardness": 4.0,
        "crystal_system": "cubic",
    },
    "apatite": {
        "name": "Apatite",
        "formula": "Ca5(PO4)3(F,Cl,OH)",
        "sg": 3.2,
        "hardness": 5.0,
        "crystal_system": "hexagonal",
    },
    "pyrite": {
        "name": "Pyrite",
        "formula": "FeS2",
        "sg": 5.02,
        "hardness": 6.0,
        "crystal_system": "cubic",
    },
    "chalcopyrite": {
        "name": "Chalcopyrite",
        "formula": "CuFeS2",
        "sg": 4.2,
        "hardness": 3.5,
        "crystal_system": "tetragonal",
    },
    "galena": {
        "name": "Galena",
        "formula": "PbS",
        "sg": 7.6,
        "hardness": 2.5,
        "crystal_system": "cubic",
    },
    "sphalerite": {
        "name": "Sphalerite",
        "formula": "ZnS",
        "sg": 4.0,
        "hardness": 3.5,
        "crystal_system": "cubic",
    },
    "magnetite": {
        "name": "Magnetite",
        "formula": "Fe3O4",
        "sg": 5.18,
        "hardness": 6.0,
        "crystal_system": "cubic",
    },
    "hematite": {
        "name": "Hematite",
        "formula": "Fe2O3",
        "sg": 5.26,
        "hardness": 5.5,
        "crystal_system": "hexagonal",
    },
    "goethite": {
        "name": "Goethite",
        "formula": "FeO(OH)",
        "sg": 3.8,
        "hardness": 5.0,
        "crystal_system": "orthorhombic",
    },
    "limonite": {
        "name": "Limonite",
        "formula": "FeO(OH)·nH2O",
        "sg": 3.6,
        "hardness": 4.0,
        "crystal_system": "amorphous",
    },
    "bauxite": {
        "name": "Bauxite",
        "formula": "Al(OH)3",
        "sg": 2.45,
        "hardness": 2.5,
        "crystal_system": "amorphous",
    },
    "corundum": {
        "name": "Corundum",
        "formula": "Al2O3",
        "sg": 4.0,
        "hardness": 9.0,
        "crystal_system": "hexagonal",
    },
    "garnet": {
        "name": "Garnet",
        "formula": "(Fe,Mg,Ca)3(Al,Fe)2(SiO4)3",
        "sg": 3.8,
        "hardness": 7.0,
        "crystal_system": "cubic",
    },
    "topaz": {
        "name": "Topaz",
        "formula": "Al2SiO4(F,OH)2",
        "sg": 3.5,
        "hardness": 8.0,
        "crystal_system": "orthorhombic",
    },
    "tourmaline": {
        "name": "Tourmaline",
        "formula": "Na(Mg,Fe)3Al6(BO3)3Si6O18(OH)4",
        "sg": 3.1,
        "hardness": 7.0,
        "crystal_system": "hexagonal",
    },
    "talc": {
        "name": "Talc",
        "formula": "Mg3Si4O10(OH)2",
        "sg": 2.75,
        "hardness": 1.0,
        "crystal_system": "monoclinic",
    },
    "kaolinite": {
        "name": "Kaolinite",
        "formula": "Al2Si2O5(OH)4",
        "sg": 2.6,
        "hardness": 2.0,
        "crystal_system": "triclinic",
    },
    "montmorillonite": {
        "name": "Montmorillonite",
        "formula": "(Na,Ca)0.3(Al,Mg)2Si4O10(OH)2·nH2O",
        "sg": 2.35,
        "hardness": 1.5,
        "crystal_system": "monoclinic",
    },
    "illite": {
        "name": "Illite",
        "formula": "K0.65Al2(Al0.65Si3.35O10)(OH)2",
        "sg": 2.75,
        "hardness": 1.5,
        "crystal_system": "monoclinic",
    },
    "chlorite": {
        "name": "Chlorite",
        "formula": "(Mg,Fe)3(Si,Al)4O10(OH)2·(Mg,Fe)3(OH)6",
        "sg": 2.7,
        "hardness": 2.0,
        "crystal_system": "monoclinic",
    },
    "serpentine": {
        "name": "Serpentine",
        "formula": "Mg3Si2O5(OH)4",
        "sg": 2.55,
        "hardness": 3.0,
        "crystal_system": "monoclinic",
    },
    "gold": {
        "name": "Gold",
        "formula": "Au",
        "sg": 19.3,
        "hardness": 2.5,
        "crystal_system": "cubic",
    },
    "silver": {
        "name": "Silver",
        "formula": "Ag",
        "sg": 10.5,
        "hardness": 2.5,
        "crystal_system": "cubic",
    },
    "copper": {
        "name": "Copper (native)",
        "formula": "Cu",
        "sg": 8.9,
        "hardness": 2.5,
        "crystal_system": "cubic",
    },
    "bornite": {
        "name": "Bornite",
        "formula": "Cu5FeS4",
        "sg": 5.08,
        "hardness": 3.0,
        "crystal_system": "orthorhombic",
    },
    "chalcocite": {
        "name": "Chalcocite",
        "formula": "Cu2S",
        "sg": 5.7,
        "hardness": 2.5,
        "crystal_system": "monoclinic",
    },
    "covellite": {
        "name": "Covellite",
        "formula": "CuS",
        "sg": 4.68,
        "hardness": 1.5,
        "crystal_system": "hexagonal",
    },
    "malachite": {
        "name": "Malachite",
        "formula": "Cu2CO3(OH)2",
        "sg": 3.9,
        "hardness": 3.5,
        "crystal_system": "monoclinic",
    },
    "azurite": {
        "name": "Azurite",
        "formula": "Cu3(CO3)2(OH)2",
        "sg": 3.77,
        "hardness": 3.5,
        "crystal_system": "monoclinic",
    },
    "molybdenite": {
        "name": "Molybdenite",
        "formula": "MoS2",
        "sg": 4.7,
        "hardness": 1.5,
        "crystal_system": "hexagonal",
    },
    "cassiterite": {
        "name": "Cassiterite",
        "formula": "SnO2",
        "sg": 6.95,
        "hardness": 6.5,
        "crystal_system": "tetragonal",
    },
    "wolframite": {
        "name": "Wolframite",
        "formula": "(Fe,Mn)WO4",
        "sg": 7.3,
        "hardness": 4.5,
        "crystal_system": "monoclinic",
    },
    "scheelite": {
        "name": "Scheelite",
        "formula": "CaWO4",
        "sg": 6.1,
        "hardness": 4.5,
        "crystal_system": "tetragonal",
    },
    "chromite": {
        "name": "Chromite",
        "formula": "FeCr2O4",
        "sg": 4.6,
        "hardness": 5.5,
        "crystal_system": "cubic",
    },
    "ilmenite": {
        "name": "Ilmenite",
        "formula": "FeTiO3",
        "sg": 4.72,
        "hardness": 5.5,
        "crystal_system": "hexagonal",
    },
    "rutile": {
        "name": "Rutile",
        "formula": "TiO2",
        "sg": 4.25,
        "hardness": 6.0,
        "crystal_system": "tetragonal",
    },
    "zircon": {
        "name": "Zircon",
        "formula": "ZrSiO4",
        "sg": 4.65,
        "hardness": 7.5,
        "crystal_system": "tetragonal",
    },
    "barite": {
        "name": "Barite",
        "formula": "BaSO4",
        "sg": 4.5,
        "hardness": 3.0,
        "crystal_system": "orthorhombic",
    },
    "wollastonite": {
        "name": "Wollastonite",
        "formula": "CaSiO3",
        "sg": 2.9,
        "hardness": 4.5,
        "crystal_system": "triclinic",
    },
    "diamond": {
        "name": "Diamond",
        "formula": "C",
        "sg": 3.52,
        "hardness": 10.0,
        "crystal_system": "cubic",
    },
    "graphite": {
        "name": "Graphite",
        "formula": "C",
        "sg": 2.2,
        "hardness": 1.5,
        "crystal_system": "hexagonal",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_mineral(name: str) -> dict[str, Any] | None:
    """Look up a mineral by name (case-insensitive).

    Parameters
    ----------
    name : str
        Mineral name, e.g. ``'Quartz'`` or ``'quartz'``.

    Returns
    -------
    dict or None
        A dictionary with keys ``name``, ``formula``, ``sg``,
        ``hardness``, ``crystal_system``; or ``None`` if not found.

    Examples
    --------
    >>> get_mineral('Pyrite')['sg']
    5.02
    >>> get_mineral('Unknown') is None
    True

    References
    ----------
    .. [1] Dana's New Mineralogy, 8th Edition, Wiley, 1997.
    """
    return MINERAL_DB.get(name.lower().strip())


def get_sg(mineral_name: str) -> float | None:
    """Return the specific gravity of a mineral.

    Parameters
    ----------
    mineral_name : str
        Mineral name (case-insensitive).

    Returns
    -------
    float or None
        Specific gravity, or ``None`` if the mineral is not in the
        database.

    Examples
    --------
    >>> get_sg('Galena')
    7.6
    >>> get_sg('Unknown') is None
    True

    References
    ----------
    .. [1] Dana's New Mineralogy, 8th Edition, Wiley, 1997.
    """
    mineral = get_mineral(mineral_name)
    if mineral is None:
        return None
    return mineral["sg"]


def search_minerals(query: str) -> list[dict[str, Any]]:
    """Search minerals by name or formula substring.

    Parameters
    ----------
    query : str
        Case-insensitive substring to match against the mineral name
        or chemical formula.

    Returns
    -------
    list of dict
        List of matching mineral entries.

    Examples
    --------
    >>> results = search_minerals('Cu')
    >>> len(results) > 0
    True

    References
    ----------
    .. [1] Dana's New Mineralogy, 8th Edition, Wiley, 1997.
    """
    query_lower = query.lower().strip()
    results: list[dict[str, Any]] = []
    for mineral in MINERAL_DB.values():
        if query_lower in mineral["name"].lower() or query_lower in mineral["formula"].lower():
            results.append(mineral)
    return results
