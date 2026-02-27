"""Sample compositing routines for drillhole assay data.

Compositing converts irregularly-spaced assay intervals into regular
composites suitable for geostatistical analysis.  All grade averaging is
**length-weighted** to honour the support volume of each sample.

References
----------
Deutsch, C. V. (2002). *Geostatistical Reservoir Modeling*. Oxford
University Press. pp. 32--35.

Isaaks, E. H. & Srivastava, R. M. (1989). *Applied Geostatistics*.
Oxford University Press. Chapter 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _detect_grade_cols(df: pd.DataFrame, grade_cols: list[str] | None) -> list[str]:
    """Return grade column names, auto-detecting if *grade_cols* is ``None``.

    Parameters
    ----------
    df : pd.DataFrame
        Assay DataFrame.
    grade_cols : list[str] or None
        Explicit list, or ``None`` for auto-detection.

    Returns
    -------
    list[str]
    """
    if grade_cols is not None:
        return list(grade_cols)

    exclude = {"hole_id", "from_depth", "to_depth", "depth", "z", "x", "y", "max_depth"}
    result = []
    for c in df.columns:
        if c in exclude:
            continue
        try:
            if np.issubdtype(df[c].dtype, np.number):
                result.append(c)
        except TypeError:
            # StringDtype and other extension types are not numeric
            pass
    return result


def composite_by_length(
    assays_df: pd.DataFrame,
    length: float,
    grade_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create equal-length composites along each drillhole.

    Partial intervals at composite boundaries are handled by
    length-weighted averaging: the grade contribution of each source
    sample is proportional to the fraction of its length that falls
    inside the target composite window.

    Parameters
    ----------
    assays_df : pd.DataFrame
        Must contain columns ``hole_id``, ``from_depth``, ``to_depth``
        and one or more numeric grade columns.
    length : float
        Target composite length (same depth units as the input).
    grade_cols : list[str], optional
        Grade columns to composite.  Detected automatically if ``None``.

    Returns
    -------
    pd.DataFrame
        Composited data with columns ``hole_id``, ``from_depth``,
        ``to_depth``, plus the composited grade columns.

    Examples
    --------
    >>> import pandas as pd
    >>> assays = pd.DataFrame({
    ...     "hole_id": ["DH1"] * 5,
    ...     "from_depth": [0, 2, 4, 6, 8],
    ...     "to_depth": [2, 4, 6, 8, 10],
    ...     "au": [1.0, 2.0, 3.0, 4.0, 5.0],
    ... })
    >>> comp = composite_by_length(assays, 10.0)
    >>> float(comp["au"].iloc[0])
    3.0
    """
    gcols = _detect_grade_cols(assays_df, grade_cols)
    results: list[dict] = []

    for hole_id, group in assays_df.groupby("hole_id", sort=False):
        group = group.sort_values("from_depth").reset_index(drop=True)
        hole_from = float(group["from_depth"].min())
        hole_to = float(group["to_depth"].max())

        comp_start = hole_from
        while comp_start < hole_to - 1e-9:
            comp_end = min(comp_start + length, hole_to)
            weighted_grades = {c: 0.0 for c in gcols}
            total_weight = 0.0

            for _, row in group.iterrows():
                s_from = float(row["from_depth"])
                s_to = float(row["to_depth"])
                # Overlap between sample and composite
                overlap_start = max(s_from, comp_start)
                overlap_end = min(s_to, comp_end)
                overlap = overlap_end - overlap_start
                if overlap <= 0:
                    continue

                total_weight += overlap
                for c in gcols:
                    val = row[c]
                    if pd.notna(val):
                        weighted_grades[c] += float(val) * overlap

            record: dict = {
                "hole_id": hole_id,
                "from_depth": comp_start,
                "to_depth": comp_end,
            }
            for c in gcols:
                if total_weight > 0:
                    record[c] = weighted_grades[c] / total_weight
                else:
                    record[c] = np.nan
            results.append(record)
            comp_start = comp_end

    return pd.DataFrame(results)


def composite_by_geology(
    assays_df: pd.DataFrame,
    geology_col: str,
    grade_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Composite assay intervals grouped by geology unit.

    Within each (hole, geology-unit) group, grades are averaged using
    length-weighting.

    Parameters
    ----------
    assays_df : pd.DataFrame
        Must include ``hole_id``, ``from_depth``, ``to_depth``, and a
        column identified by *geology_col*.
    geology_col : str
        Name of the column containing geology codes/lithologies.
    grade_cols : list[str], optional
        Grade columns to composite.  Detected automatically if ``None``.

    Returns
    -------
    pd.DataFrame
        One row per (hole, geology-unit) with length-weighted averages.
    """
    gcols = _detect_grade_cols(assays_df, grade_cols)
    results: list[dict] = []

    for (hole_id, geo_unit), group in assays_df.groupby(["hole_id", geology_col], sort=False):
        group = group.sort_values("from_depth").reset_index(drop=True)
        lengths = (group["to_depth"] - group["from_depth"]).astype(float)
        total_len = lengths.sum()

        record: dict = {
            "hole_id": hole_id,
            geology_col: geo_unit,
            "from_depth": float(group["from_depth"].min()),
            "to_depth": float(group["to_depth"].max()),
        }
        for c in gcols:
            vals = group[c].astype(float)
            mask = vals.notna()
            if mask.any() and total_len > 0:
                record[c] = float((vals[mask] * lengths[mask]).sum() / lengths[mask].sum())
            else:
                record[c] = np.nan
        results.append(record)

    return pd.DataFrame(results)


def composite_by_bench(
    assays_df: pd.DataFrame,
    bench_height: float,
    collar_z_col: str = "z",
    grade_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create bench composites based on elevation intervals.

    Each bench is defined by a fixed vertical height.  The elevation of
    each sample interval is derived from the collar elevation (``z``)
    minus the midpoint depth.  Grades are length-weighted within each
    bench.

    Parameters
    ----------
    assays_df : pd.DataFrame
        Must include ``hole_id``, ``from_depth``, ``to_depth``, and a
        column with collar elevation (default ``z``).
    bench_height : float
        Vertical height of each bench (positive value).
    collar_z_col : str, optional
        Name of the column containing collar elevation (default ``"z"``).
    grade_cols : list[str], optional
        Grade columns to composite.  Detected automatically if ``None``.

    Returns
    -------
    pd.DataFrame
        Composited data with ``hole_id``, ``bench_top``, ``bench_bottom``,
        and length-weighted grade columns.
    """
    gcols = _detect_grade_cols(assays_df, grade_cols)
    results: list[dict] = []

    for hole_id, group in assays_df.groupby("hole_id", sort=False):
        group = group.sort_values("from_depth").reset_index(drop=True)
        collar_z = float(group[collar_z_col].iloc[0])

        # Compute elevation of interval top and bottom
        elev_top = collar_z - group["from_depth"].astype(float)
        elev_bottom = collar_z - group["to_depth"].astype(float)

        # Determine bench boundaries
        min_elev = elev_bottom.min()
        max_elev = elev_top.max()

        # Bench tops descending from ceiling
        bench_ceil = np.ceil(max_elev / bench_height) * bench_height
        bench_floor = np.floor(min_elev / bench_height) * bench_height

        bench_top = bench_ceil
        while bench_top > bench_floor:
            bench_bot = bench_top - bench_height
            weighted_grades = {c: 0.0 for c in gcols}
            total_weight = 0.0

            for idx, row in group.iterrows():
                s_etop = float(elev_top.loc[idx])
                s_ebot = float(elev_bottom.loc[idx])
                # Overlap in elevation space
                overlap_top = min(s_etop, bench_top)
                overlap_bot = max(s_ebot, bench_bot)
                overlap = overlap_top - overlap_bot
                if overlap <= 0:
                    continue

                total_weight += overlap
                for c in gcols:
                    val = row[c]
                    if pd.notna(val):
                        weighted_grades[c] += float(val) * overlap

            if total_weight > 0:
                record: dict = {
                    "hole_id": hole_id,
                    "bench_top": bench_top,
                    "bench_bottom": bench_bot,
                }
                for c in gcols:
                    record[c] = weighted_grades[c] / total_weight
                results.append(record)

            bench_top = bench_bot

    return pd.DataFrame(results)
