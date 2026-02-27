"""Rock blastability index and rock factor estimation.

This module implements the Lilly Blastability Index (BI) and its
conversion to the Cunningham rock factor *A* used in the Kuz-Ram
fragmentation model.

References
----------
.. [1] Lilly, P.A. (1986). An empirical method of assessing rock mass
       blastability. *Proc. Large Open Pit Mining Conf.*, AusIMM,
       Melbourne, 89-92.
.. [2] Cunningham, C.V.B. (1987). Fragmentation estimations and the
       Kuz-Ram model — four years on. *Proc. 2nd Int. Symp. on Rock
       Fragmentation by Blasting*, 475-487.
"""

from __future__ import annotations

from minelab.utilities.validators import validate_positive, validate_range

# ---------------------------------------------------------------------------
# Lilly Blastability Index
# ---------------------------------------------------------------------------


def lilly_blastability_index(
    rmd: float,
    jf: float,
    jps: float,
    rdi: float,
    hf: float,
) -> float:
    """Compute the Lilly Blastability Index (BI).

    .. math::

        BI = 0.5 \\, (RMD + JF + JPS + RDI + HF)

    Each sub-rating is assigned from field observation:

    - **RMD** — Rock Mass Description: powdery/friable = 10,
      blocky = 20, totally massive = 50.
    - **JF** — Joint Factor (joint plane spacing + orientation):
      typical range 10 to 50.
    - **JPS** — Joint Plane Spacing: close (< 0.1 m) = 10,
      intermediate = 20, wide (> 1 m) = 50.
    - **RDI** — Rock Density Influence: 25 * (density - 50/density),
      typical 5 to 50 depending on SG.
    - **HF** — Hardness Factor: from UCS/Young's modulus, range 1 to 10
      but scaled here in the same order (typical 1 to 50).

    Parameters
    ----------
    rmd : float
        Rock Mass Description rating (10 to 50).
    jf : float
        Joint Factor rating (10 to 50).
    jps : float
        Joint Plane Spacing rating (10 to 50).
    rdi : float
        Rock Density Influence rating (5 to 50).
    hf : float
        Hardness Factor rating (1 to 50).

    Returns
    -------
    float
        Blastability Index (dimensionless). Typical range ~20 to 100.

    Examples
    --------
    >>> lilly_blastability_index(20, 30, 20, 25, 10)
    52.5

    References
    ----------
    .. [1] Lilly (1986).
    """
    validate_range(rmd, 10, 50, "rmd")
    validate_range(jf, 10, 50, "jf")
    validate_range(jps, 10, 50, "jps")
    validate_range(rdi, 5, 50, "rdi")
    validate_range(hf, 1, 50, "hf")

    return 0.5 * (rmd + jf + jps + rdi + hf)


# ---------------------------------------------------------------------------
# Rock Factor from Blastability Index
# ---------------------------------------------------------------------------


def rock_factor_from_bi(bi: float) -> float:
    """Convert a Lilly Blastability Index to a Kuz-Ram rock factor *A*.

    .. math::

        A = 0.06 \\, BI

    This simplified relationship (Cunningham 1987) converts the
    dimensionless blastability index into the rock factor used in the
    Kuz-Ram fragmentation model.

    Parameters
    ----------
    bi : float
        Blastability Index. Must be positive.

    Returns
    -------
    float
        Rock factor *A* for use in Kuz-Ram. Typical range ~1 to 6
        for most mining conditions; can reach ~13 for very hard,
        massive rock.

    Examples
    --------
    >>> rock_factor_from_bi(52.5)
    3.15
    >>> rock_factor_from_bi(100)
    6.0

    References
    ----------
    .. [1] Cunningham (1987).
    """
    validate_positive(bi, "bi")

    return 0.06 * bi
