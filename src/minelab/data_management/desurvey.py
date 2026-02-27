"""Drillhole desurvey methods.

Convert downhole survey measurements (depth, azimuth, dip) into 3-D
Cartesian coordinates.  Three standard interpolation algorithms are
provided:

* **Minimum curvature** -- industry standard; honours curvature.
* **Tangential** -- uses bottom-of-interval orientation.
* **Balanced tangential** -- average of top and bottom orientations.

Mining dip convention
---------------------
Dip is measured from horizontal.  A dip of -90 degrees points
vertically downward; 0 is horizontal; +90 is vertically upward.
Internally the functions convert dip to the inclination angle measured
from the horizontal before applying trigonometric functions.

References
----------
Saúl, M. (2012). *Minimum curvature method applied to drillholes*.
Technical Note, SRK Consulting.

Devico AS (2019). Directional drilling -- Survey calculation methods.
Technical Report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dip_azi_to_rad(dip_deg: np.ndarray, azi_deg: np.ndarray):
    """Convert mining-convention dip and azimuth to radians.

    Parameters
    ----------
    dip_deg : array-like
        Dip from horizontal in degrees (negative = downward).
    azi_deg : array-like
        Azimuth in degrees from north, clockwise.

    Returns
    -------
    inc_rad : np.ndarray
        Inclination from horizontal in radians (positive downward for
        negative dip input).
    azi_rad : np.ndarray
        Azimuth in radians.

    Notes
    -----
    Mining convention: dip = -90 is straight down.
    For the trig in desurvey we need the angle from horizontal with
    the sign preserved (sin(-90 deg) = -1 gives downward component).
    """
    inc_rad = np.deg2rad(dip_deg.astype(float))
    azi_rad = np.deg2rad(azi_deg.astype(float))
    return inc_rad, azi_rad


def minimum_curvature(surveys_df: pd.DataFrame) -> pd.DataFrame:
    """Desurvey using the minimum-curvature method.

    For each survey interval the dogleg angle is computed and used to
    derive a ratio factor (RF).  The incremental displacements are:

    .. math::

        dx = \\frac{\\Delta MD}{2}
             (\\sin I_1 \\sin A_1 + \\sin I_2 \\sin A_2) \\times RF

        dy = \\frac{\\Delta MD}{2}
             (\\sin I_1 \\cos A_1 + \\sin I_2 \\cos A_2) \\times RF

        dz = \\frac{\\Delta MD}{2}
             (\\cos I_1 + \\cos I_2) \\times RF

    where *I* is the inclination from vertical (= 90 + dip) and *A* is
    the azimuth.

    Parameters
    ----------
    surveys_df : pd.DataFrame
        Must contain columns ``hole_id``, ``depth``, ``azimuth``, ``dip``.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``dx``, ``dy``, ``dz`` (cumulative offsets
        from collar) and intermediate ``delta_md``.

    Notes
    -----
    Dip convention: negative = downward from horizontal.
    The inclination from vertical (used in standard formulas) is
    ``incl = 90 + dip``.  So a vertical-down hole (dip = -90) gives
    ``incl = 0`` (aligned with vertical axis).
    """
    results: list[pd.DataFrame] = []

    for _hole_id, grp in surveys_df.groupby("hole_id", sort=False):
        grp = grp.sort_values("depth").reset_index(drop=True)
        n = len(grp)
        depths = grp["depth"].values.astype(float)
        dips = grp["dip"].values.astype(float)
        azis = grp["azimuth"].values.astype(float)

        # Convert dip (from horizontal, -90 = down) to inclination from
        # vertical: incl = 90 + dip.  dip=-90 => incl=0 (vertical down).
        incl = np.deg2rad(90.0 + dips)  # inclination from vertical
        azi = np.deg2rad(azis)

        dx_cum = np.zeros(n)
        dy_cum = np.zeros(n)
        dz_cum = np.zeros(n)

        for i in range(1, n):
            dmd = depths[i] - depths[i - 1]
            if dmd <= 0:
                continue

            i1, a1 = incl[i - 1], azi[i - 1]
            i2, a2 = incl[i], azi[i]

            # Dogleg angle
            cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
            # Clamp for numerical safety
            cos_dl = np.clip(cos_dl, -1.0, 1.0)
            dogleg = np.arccos(cos_dl)

            # Ratio factor
            rf = (2.0 / dogleg) * np.tan(dogleg / 2.0) if dogleg > 1e-7 else 1.0

            # Incremental offsets  (x=East, y=North, z=Down)
            dx = (dmd / 2.0) * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
            dy = (dmd / 2.0) * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
            dz = (dmd / 2.0) * (np.cos(i1) + np.cos(i2)) * rf

            dx_cum[i] = dx_cum[i - 1] + dx
            dy_cum[i] = dy_cum[i - 1] + dy
            dz_cum[i] = dz_cum[i - 1] + dz

        out = grp.copy()
        out["dx"] = dx_cum
        out["dy"] = dy_cum
        out["dz"] = dz_cum
        results.append(out)

    if not results:
        return pd.DataFrame(columns=list(surveys_df.columns) + ["dx", "dy", "dz"])
    return pd.concat(results, ignore_index=True)


def tangential(surveys_df: pd.DataFrame) -> pd.DataFrame:
    """Desurvey using the simple tangential method.

    Each interval uses the orientation at the **bottom** survey station
    to compute the displacement over the entire interval length.

    Parameters
    ----------
    surveys_df : pd.DataFrame
        Must contain ``hole_id``, ``depth``, ``azimuth``, ``dip``.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``dx``, ``dy``, ``dz``.
    """
    results: list[pd.DataFrame] = []

    for _hole_id, grp in surveys_df.groupby("hole_id", sort=False):
        grp = grp.sort_values("depth").reset_index(drop=True)
        n = len(grp)
        depths = grp["depth"].values.astype(float)
        dips = grp["dip"].values.astype(float)
        azis = grp["azimuth"].values.astype(float)

        incl = np.deg2rad(90.0 + dips)
        azi = np.deg2rad(azis)

        dx_cum = np.zeros(n)
        dy_cum = np.zeros(n)
        dz_cum = np.zeros(n)

        for i in range(1, n):
            dmd = depths[i] - depths[i - 1]
            if dmd <= 0:
                continue
            # Use bottom-of-interval orientation
            dx_cum[i] = dx_cum[i - 1] + dmd * np.sin(incl[i]) * np.sin(azi[i])
            dy_cum[i] = dy_cum[i - 1] + dmd * np.sin(incl[i]) * np.cos(azi[i])
            dz_cum[i] = dz_cum[i - 1] + dmd * np.cos(incl[i])

        out = grp.copy()
        out["dx"] = dx_cum
        out["dy"] = dy_cum
        out["dz"] = dz_cum
        results.append(out)

    if not results:
        return pd.DataFrame(columns=list(surveys_df.columns) + ["dx", "dy", "dz"])
    return pd.concat(results, ignore_index=True)


def balanced_tangential(surveys_df: pd.DataFrame) -> pd.DataFrame:
    """Desurvey using the balanced tangential method.

    The displacement for each interval is the simple average of the
    tangential vectors computed at the top and bottom stations.

    Parameters
    ----------
    surveys_df : pd.DataFrame
        Must contain ``hole_id``, ``depth``, ``azimuth``, ``dip``.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``dx``, ``dy``, ``dz``.
    """
    results: list[pd.DataFrame] = []

    for _hole_id, grp in surveys_df.groupby("hole_id", sort=False):
        grp = grp.sort_values("depth").reset_index(drop=True)
        n = len(grp)
        depths = grp["depth"].values.astype(float)
        dips = grp["dip"].values.astype(float)
        azis = grp["azimuth"].values.astype(float)

        incl = np.deg2rad(90.0 + dips)
        azi = np.deg2rad(azis)

        dx_cum = np.zeros(n)
        dy_cum = np.zeros(n)
        dz_cum = np.zeros(n)

        for i in range(1, n):
            dmd = depths[i] - depths[i - 1]
            if dmd <= 0:
                continue

            i1, a1 = incl[i - 1], azi[i - 1]
            i2, a2 = incl[i], azi[i]

            dx_top = dmd * np.sin(i1) * np.sin(a1)
            dy_top = dmd * np.sin(i1) * np.cos(a1)
            dz_top = dmd * np.cos(i1)

            dx_bot = dmd * np.sin(i2) * np.sin(a2)
            dy_bot = dmd * np.sin(i2) * np.cos(a2)
            dz_bot = dmd * np.cos(i2)

            dx_cum[i] = dx_cum[i - 1] + (dx_top + dx_bot) / 2.0
            dy_cum[i] = dy_cum[i - 1] + (dy_top + dy_bot) / 2.0
            dz_cum[i] = dz_cum[i - 1] + (dz_top + dz_bot) / 2.0

        out = grp.copy()
        out["dx"] = dx_cum
        out["dy"] = dy_cum
        out["dz"] = dz_cum
        results.append(out)

    if not results:
        return pd.DataFrame(columns=list(surveys_df.columns) + ["dx", "dy", "dz"])
    return pd.concat(results, ignore_index=True)


def compute_coordinates(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    desurvey_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add collar coordinates to desurvey offsets to obtain absolute XYZ.

    The ``dz`` column from the desurvey represents downward displacement.
    Since collar *z* is elevation (positive up), the absolute elevation is
    ``collar_z - dz``.

    Parameters
    ----------
    collar_x, collar_y, collar_z : float
        Collar easting, northing, and elevation.
    desurvey_df : pd.DataFrame
        Output from one of the desurvey functions.  Must contain
        ``dx``, ``dy``, ``dz``.

    Returns
    -------
    pd.DataFrame
        Copy of *desurvey_df* with added columns ``x``, ``y``, ``z``
        representing absolute coordinates.
    """
    out = desurvey_df.copy()
    out["x"] = collar_x + out["dx"].astype(float)
    out["y"] = collar_y + out["dy"].astype(float)
    out["z"] = collar_z - out["dz"].astype(float)  # z is elevation, dz is downward
    return out
