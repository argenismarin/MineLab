"""NI 43-101 resource classification (Canadian standard).

This module implements block-level resource classification following the
Canadian Institute of Mining (CIM) 2014 Definition Standards as required
by National Instrument 43-101.

References
----------
.. [1] CIM (2014). *CIM Definition Standards for Mineral Resources and
       Mineral Reserves*. Canadian Institute of Mining, Metallurgy and
       Petroleum.
.. [2] NI 43-101 (2011). *Standards of Disclosure for Mineral Projects*.
       Canadian Securities Administrators.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Default Thresholds
# ---------------------------------------------------------------------------

_DEFAULT_NI43101_THRESHOLDS: dict = {
    "measured": {
        "spacing_max": 25.0,
        "continuity_min": 0.8,
        "confidence_min": 0.9,
    },
    "indicated": {
        "spacing_max": 50.0,
        "continuity_min": 0.5,
        "confidence_min": 0.7,
    },
}


# ---------------------------------------------------------------------------
# NI 43-101 Classification
# ---------------------------------------------------------------------------


def ni43101_classify(
    data_spacing: np.ndarray,
    continuity: np.ndarray,
    confidence: np.ndarray,
    thresholds: dict | None = None,
) -> np.ndarray:
    """Classify resource blocks according to NI 43-101 / CIM 2014.

    Each block is assigned a category based on data spacing, geological
    continuity, and estimation confidence:

    - **Measured** (1): spacing <= spacing_max AND continuity >=
      continuity_min AND confidence >= confidence_min.
    - **Indicated** (2): same logic with relaxed thresholds.
    - **Inferred** (3): all remaining blocks.

    Parameters
    ----------
    data_spacing : numpy.ndarray
        1-D array of average drillhole spacing per block (metres).
    continuity : numpy.ndarray
        1-D array of geological continuity indices per block, typically
        in [0, 1] (0 = no continuity, 1 = perfect continuity).
    confidence : numpy.ndarray
        1-D array of estimation confidence values per block, typically
        in [0, 1] (e.g. derived from kriging efficiency or cross-
        validation).
    thresholds : dict or None, optional
        Nested dictionary defining classification thresholds. If ``None``,
        default thresholds are used::

            {
                "measured": {
                    "spacing_max": 25,
                    "continuity_min": 0.8,
                    "confidence_min": 0.9
                },
                "indicated": {
                    "spacing_max": 50,
                    "continuity_min": 0.5,
                    "confidence_min": 0.7
                }
            }

    Returns
    -------
    numpy.ndarray
        1-D integer array of category codes: 1 = Measured, 2 = Indicated,
        3 = Inferred.

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty, or if
        required threshold keys are missing.

    Examples
    --------
    >>> import numpy as np
    >>> spacing = np.array([20, 40, 80, 22, 55])
    >>> cont = np.array([0.9, 0.6, 0.3, 0.85, 0.4])
    >>> conf = np.array([0.95, 0.75, 0.4, 0.92, 0.5])
    >>> ni43101_classify(spacing, cont, conf)
    array([1, 2, 3, 1, 3])

    References
    ----------
    .. [1] CIM (2014), Definition Standards.
    .. [2] NI 43-101 (2011), Part 6.
    """
    ds = np.asarray(data_spacing, dtype=float).ravel()
    cont = np.asarray(continuity, dtype=float).ravel()
    conf = np.asarray(confidence, dtype=float).ravel()

    if ds.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (ds.size == cont.size == conf.size):
        raise ValueError(
            f"All input arrays must have the same length. Got "
            f"data_spacing={ds.size}, continuity={cont.size}, "
            f"confidence={conf.size}."
        )

    if thresholds is None:
        thresholds = _DEFAULT_NI43101_THRESHOLDS

    for cat in ("measured", "indicated"):
        if cat not in thresholds:
            raise ValueError(f"thresholds must contain '{cat}' key.")
        for key in ("spacing_max", "continuity_min", "confidence_min"):
            if key not in thresholds[cat]:
                raise ValueError(f"thresholds['{cat}'] must contain '{key}' key.")

    # Start with all blocks as Inferred (3)
    classification = np.full(ds.size, 3, dtype=int)

    # Indicated (2): relaxed thresholds
    ind = thresholds["indicated"]
    indicated_mask = (
        (ds <= ind["spacing_max"])
        & (cont >= ind["continuity_min"])
        & (conf >= ind["confidence_min"])
    )
    classification[indicated_mask] = 2

    # Measured (1): strict thresholds (overrides Indicated)
    meas = thresholds["measured"]
    measured_mask = (
        (ds <= meas["spacing_max"])
        & (cont >= meas["continuity_min"])
        & (conf >= meas["confidence_min"])
    )
    classification[measured_mask] = 1

    return classification
