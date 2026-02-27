"""Data quality assurance and quality control (QA/QC) for drillhole data.

Provides validation functions that detect common data-entry and
measurement errors in collar, survey, and assay tables.

References
----------
Snowden Mining Industry Consultants (2010). *Best Practices for
Drillhole Database Management*. Technical Note.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from minelab.data_management.drillholes import DrillholeDB


def check_collar_duplicates(collars_df: pd.DataFrame) -> list[str]:
    """Find duplicate hole IDs in the collar table.

    Parameters
    ----------
    collars_df : pd.DataFrame
        Collar table with at least a ``hole_id`` column.

    Returns
    -------
    list[str]
        List of hole IDs that appear more than once.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"hole_id": ["A", "B", "A"]})
    >>> check_collar_duplicates(df)
    ['A']
    """
    duplicated_mask = collars_df["hole_id"].duplicated(keep=False)
    if not duplicated_mask.any():
        return []
    return sorted(collars_df.loc[duplicated_mask, "hole_id"].unique().tolist())


def check_survey_consistency(surveys_df: pd.DataFrame) -> list[str]:
    """Check survey data for out-of-range dip and azimuth values.

    Valid ranges:

    * **dip**: [-90, 90] degrees (mining convention)
    * **azimuth**: [0, 360) degrees

    Parameters
    ----------
    surveys_df : pd.DataFrame
        Survey table with columns ``hole_id``, ``depth``, ``azimuth``,
        ``dip``.

    Returns
    -------
    list[str]
        List of human-readable issue descriptions.
    """
    issues: list[str] = []

    for _idx, row in surveys_df.iterrows():
        dip = float(row["dip"])
        azi = float(row["azimuth"])
        hid = row["hole_id"]
        depth = row["depth"]

        if dip < -90.0 or dip > 90.0:
            issues.append(f"Hole {hid} at depth {depth}: dip {dip} out of range [-90, 90]")
        if azi < 0.0 or azi >= 360.0:
            issues.append(f"Hole {hid} at depth {depth}: azimuth {azi} out of range [0, 360)")

    return issues


def check_assay_overlaps(assays_df: pd.DataFrame) -> pd.DataFrame:
    """Find overlapping assay intervals within each drillhole.

    Two intervals overlap when the start of the later interval is less
    than the end of the earlier interval.

    Parameters
    ----------
    assays_df : pd.DataFrame
        Assay table with columns ``hole_id``, ``from_depth``, ``to_depth``.

    Returns
    -------
    pd.DataFrame
        DataFrame of overlapping pairs with columns ``hole_id``,
        ``from_depth_1``, ``to_depth_1``, ``from_depth_2``, ``to_depth_2``,
        ``overlap``.
    """
    overlaps: list[dict] = []

    for hole_id, group in assays_df.groupby("hole_id", sort=False):
        group = group.sort_values("from_depth").reset_index(drop=True)
        for i in range(len(group) - 1):
            to_curr = float(group.iloc[i]["to_depth"])
            from_next = float(group.iloc[i + 1]["from_depth"])
            if from_next < to_curr - 1e-9:
                overlap_amount = to_curr - from_next
                overlaps.append(
                    {
                        "hole_id": hole_id,
                        "from_depth_1": float(group.iloc[i]["from_depth"]),
                        "to_depth_1": to_curr,
                        "from_depth_2": from_next,
                        "to_depth_2": float(group.iloc[i + 1]["to_depth"]),
                        "overlap": overlap_amount,
                    }
                )

    return pd.DataFrame(overlaps)


def check_assay_gaps(
    assays_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Find gaps between consecutive assay intervals within each drillhole.

    A gap exists when the start of the next interval exceeds the end of
    the previous interval by more than *tolerance*.

    Parameters
    ----------
    assays_df : pd.DataFrame
        Assay table with columns ``hole_id``, ``from_depth``, ``to_depth``.
    tolerance : float, optional
        Allowable gap (in depth units) before flagging (default 0.01).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``hole_id``, ``to_depth_1``,
        ``from_depth_2``, ``gap``.
    """
    gaps: list[dict] = []

    for hole_id, group in assays_df.groupby("hole_id", sort=False):
        group = group.sort_values("from_depth").reset_index(drop=True)
        for i in range(len(group) - 1):
            to_curr = float(group.iloc[i]["to_depth"])
            from_next = float(group.iloc[i + 1]["from_depth"])
            gap = from_next - to_curr
            if gap > tolerance:
                gaps.append(
                    {
                        "hole_id": hole_id,
                        "to_depth_1": to_curr,
                        "from_depth_2": from_next,
                        "gap": gap,
                    }
                )

    return pd.DataFrame(gaps)


def validation_report(drillhole_db: DrillholeDB) -> dict:
    """Run all QA/QC checks and return a summary report.

    Parameters
    ----------
    drillhole_db : DrillholeDB
        Populated drillhole database to validate.

    Returns
    -------
    dict
        Keys:

        * ``"collar_duplicates"`` -- list of duplicate hole IDs.
        * ``"survey_issues"`` -- list of survey consistency issues.
        * ``"assay_overlaps"`` -- DataFrame of overlapping intervals.
        * ``"assay_gaps"`` -- DataFrame of gaps.
        * ``"db_validation"`` -- list from ``DrillholeDB.validate()``.
        * ``"is_valid"`` -- ``True`` if no issues were found.
    """
    collar_dups = check_collar_duplicates(drillhole_db.collars)
    survey_issues = check_survey_consistency(drillhole_db.surveys)
    overlaps = check_assay_overlaps(drillhole_db.assays)
    gaps = check_assay_gaps(drillhole_db.assays)
    db_messages = drillhole_db.validate()

    is_valid = (
        len(collar_dups) == 0
        and len(survey_issues) == 0
        and overlaps.empty
        and gaps.empty
        and len(db_messages) == 0
    )

    return {
        "collar_duplicates": collar_dups,
        "survey_issues": survey_issues,
        "assay_overlaps": overlaps,
        "assay_gaps": gaps,
        "db_validation": db_messages,
        "is_valid": is_valid,
    }
