"""Acid mine drainage prediction and classification.

This module provides functions for assessing the acid-generating potential of
mine waste materials.  It includes Maximum Potential Acidity (MPA), Acid
Neutralizing Capacity (ANC), Net Acid Producing Potential (NAPP), NAG test
classification, and empirical paste-pH prediction.

References
----------
.. [1] AMIRA (2002). ARD Test Handbook, Project P387A.
.. [2] Sobek, A.A., Schuller, W.A., Freeman, J.R. & Smith, R.M. (1978).
       Field and Laboratory Methods Applicable to Overburdens and Minesoils.
       EPA-600/2-78-054.
.. [3] Price, W.A. (2009). Prediction Manual for Drainage Chemistry from
       Sulphidic Geologic Materials. MEND Report 1.20.1.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_range,
)

# ---------------------------------------------------------------------------
# Maximum Potential Acidity
# ---------------------------------------------------------------------------


def maximum_potential_acidity(sulfur_pct: float) -> float:
    """Compute Maximum Potential Acidity from total sulfur content.

    MPA quantifies the maximum amount of sulfuric acid that could be produced
    from the complete oxidation of sulfide minerals in a sample.

    .. math::

        \\text{MPA} = \\%S \\times 30.6 \\;\\text{(kg H}_2\\text{SO}_4\\text{/t)}

    Parameters
    ----------
    sulfur_pct : float
        Total sulfur content as a weight percentage (0--100).

    Returns
    -------
    float
        Maximum potential acidity in kg H2SO4 per tonne of material.

    Examples
    --------
    >>> maximum_potential_acidity(2.0)
    61.2

    References
    ----------
    .. [1] AMIRA P387A (2002), ARD Test Handbook.
    """
    validate_non_negative(sulfur_pct, "sulfur_pct")
    validate_range(sulfur_pct, 0, 100, "sulfur_pct")
    return sulfur_pct * 30.6


# ---------------------------------------------------------------------------
# Acid Neutralizing Capacity
# ---------------------------------------------------------------------------


def acid_neutralizing_capacity(ite_data: dict) -> float:
    """Compute Acid Neutralizing Capacity by the Sobek method.

    ANC measures the capacity of a material to neutralize acid, typically
    dominated by carbonate minerals.

    Two input modes are supported:

    1. **Direct CaCO3 percentage**:
       ``ANC = CaCO3\\% \\times 10``  (kg H2SO4/t)
    2. **Calcium and magnesium percentages**:
       ``ANC = (Ca\\% \\times 2.497 + Mg\\% \\times 4.116) \\times 10``

    Parameters
    ----------
    ite_data : dict
        Dictionary with one of the following key sets:

        - ``{"calcium_carbonate_pct": float}`` -- direct CaCO3 content.
        - ``{"calcium_pct": float, "magnesium_pct": float}`` -- elemental
          Ca and Mg contents.

    Returns
    -------
    float
        Acid neutralizing capacity in kg H2SO4 per tonne of material.

    Raises
    ------
    ValueError
        If the required keys are not present in *ite_data*.

    Examples
    --------
    >>> acid_neutralizing_capacity({"calcium_carbonate_pct": 5.0})
    50.0

    >>> acid_neutralizing_capacity({"calcium_pct": 2.0, "magnesium_pct": 1.0})
    91.1

    References
    ----------
    .. [1] Sobek et al. (1978). EPA-600/2-78-054.
    .. [2] AMIRA P387A (2002), ARD Test Handbook.
    """
    if "calcium_carbonate_pct" in ite_data:
        caco3 = ite_data["calcium_carbonate_pct"]
        validate_non_negative(caco3, "calcium_carbonate_pct")
        return caco3 * 10.0
    elif "calcium_pct" in ite_data and "magnesium_pct" in ite_data:
        ca = ite_data["calcium_pct"]
        mg = ite_data["magnesium_pct"]
        validate_non_negative(ca, "calcium_pct")
        validate_non_negative(mg, "magnesium_pct")
        # Convert elemental Ca/Mg to CaCO3-equivalent, then to kg H2SO4/t
        return (ca * 2.497 + mg * 4.116) * 10.0
    else:
        raise ValueError(
            "ite_data must contain 'calcium_carbonate_pct' or both "
            "'calcium_pct' and 'magnesium_pct'."
        )


# ---------------------------------------------------------------------------
# Net Acid Producing Potential
# ---------------------------------------------------------------------------


def napp(mpa: float, anc: float) -> dict:
    """Compute Net Acid Producing Potential and classify the material.

    .. math::

        \\text{NAPP} = \\text{MPA} - \\text{ANC}

    Classification rules:

    - **PAF** (Potentially Acid Forming): NAPP > 0
    - **NAF** (Non-Acid Forming): NAPP < 0
    - **Uncertain**: NAPP == 0

    Parameters
    ----------
    mpa : float
        Maximum Potential Acidity (kg H2SO4/t).
    anc : float
        Acid Neutralizing Capacity (kg H2SO4/t).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"napp"`` : float -- Net acid producing potential.
        - ``"classification"`` : str -- ``"PAF"``, ``"NAF"``, or
          ``"Uncertain"``.

    Examples
    --------
    >>> result = napp(60.0, 30.0)
    >>> result["napp"]
    30.0
    >>> result["classification"]
    'PAF'

    >>> napp(20.0, 50.0)["classification"]
    'NAF'

    References
    ----------
    .. [1] AMIRA P387A (2002), ARD Test Handbook.
    """
    validate_non_negative(mpa, "mpa")
    validate_non_negative(anc, "anc")
    napp_value = mpa - anc

    if napp_value > 0:
        classification = "PAF"
    elif napp_value < 0:
        classification = "NAF"
    else:
        classification = "Uncertain"

    return {"napp": napp_value, "classification": classification}


# ---------------------------------------------------------------------------
# NAG Test Classification
# ---------------------------------------------------------------------------


def nag_test_classify(nag_ph: float, nag_value: float) -> dict:
    """Classify material using Net Acid Generation (NAG) test results.

    Classification rules:

    - **PAF**: NAG pH < 4.5 **and** NAG value > 5 kg H2SO4/t
    - **NAF**: NAG pH >= 4.5
    - **Uncertain**: NAG pH < 4.5 but NAG value <= 5

    Parameters
    ----------
    nag_ph : float
        Final pH of the NAG test solution (0--14).
    nag_value : float
        NAG value in kg H2SO4 per tonne.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"classification"`` : str -- ``"PAF"``, ``"NAF"``, or
          ``"Uncertain"``.
        - ``"nag_ph"`` : float -- Input NAG pH.
        - ``"nag_value"`` : float -- Input NAG value.

    Examples
    --------
    >>> nag_test_classify(3.5, 10.0)["classification"]
    'PAF'

    >>> nag_test_classify(5.0, 2.0)["classification"]
    'NAF'

    >>> nag_test_classify(4.0, 3.0)["classification"]
    'Uncertain'

    References
    ----------
    .. [1] AMIRA P387A (2002), ARD Test Handbook, Table 5.1.
    """
    validate_range(nag_ph, 0, 14, "nag_ph")
    validate_non_negative(nag_value, "nag_value")

    if nag_ph >= 4.5:
        classification = "NAF"
    elif nag_value > 5.0:
        classification = "PAF"
    else:
        classification = "Uncertain"

    return {
        "classification": classification,
        "nag_ph": nag_ph,
        "nag_value": nag_value,
    }


# ---------------------------------------------------------------------------
# Paste pH Prediction
# ---------------------------------------------------------------------------


def paste_ph_prediction(
    sulfide_pct: float,
    neutralizer_pct: float = 0.0,
) -> dict:
    """Estimate paste pH from sulfide and neutralizer content.

    A simplified empirical relationship for screening-level prediction:

    .. math::

        \\text{pH} \\approx 7 - 3 \\times \\%\\text{sulfide}
        + 2 \\times \\%\\text{neutralizer}

    The result is clamped to the valid pH range [1, 14].

    Parameters
    ----------
    sulfide_pct : float
        Sulfide mineral content as a weight percentage (0--100).
    neutralizer_pct : float, optional
        Neutralizer (e.g. CaCO3) content as a weight percentage (0--100).
        Default is 0.0.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"predicted_ph"`` : float -- Estimated paste pH (1--14).
        - ``"classification"`` : str -- ``"Acidic"`` (pH < 5.5),
          ``"Neutral"`` (5.5 <= pH <= 8.5), or ``"Alkaline"`` (pH > 8.5).

    Examples
    --------
    >>> result = paste_ph_prediction(2.0, 0.0)
    >>> result["predicted_ph"]
    1.0
    >>> result["classification"]
    'Acidic'

    >>> paste_ph_prediction(0.5, 1.0)["predicted_ph"]
    7.5

    References
    ----------
    .. [1] Price, W.A. (2009). Prediction Manual for Drainage Chemistry
           from Sulphidic Geologic Materials. MEND Report 1.20.1.
    """
    validate_non_negative(sulfide_pct, "sulfide_pct")
    validate_range(sulfide_pct, 0, 100, "sulfide_pct")
    validate_non_negative(neutralizer_pct, "neutralizer_pct")
    validate_range(neutralizer_pct, 0, 100, "neutralizer_pct")

    raw_ph = 7.0 - 3.0 * sulfide_pct + 2.0 * neutralizer_pct
    predicted_ph = float(max(1.0, min(14.0, raw_ph)))

    if predicted_ph < 5.5:
        classification = "Acidic"
    elif predicted_ph > 8.5:
        classification = "Alkaline"
    else:
        classification = "Neutral"

    return {"predicted_ph": predicted_ph, "classification": classification}
