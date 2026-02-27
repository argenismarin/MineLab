"""Drillhole database for managing collar, survey, and assay data.

Provides the :class:`DrillholeDB` container that stores collar locations,
downhole survey measurements, and assay intervals, with built-in validation
and merging utilities.

References
----------
Hustrulid, W., Kuchta, M. & Martin, R. (2013). *Open Pit Mine Planning
and Design*, 3rd ed.  CRC Press. Chapter 4 -- Drillhole Databases.
"""

from __future__ import annotations

import pandas as pd


class DrillholeDB:
    """In-memory database for drillhole collar, survey, and assay data.

    Parameters
    ----------
    None

    Attributes
    ----------
    collars : pd.DataFrame
        Collar table with columns ``hole_id, x, y, z, max_depth``.
    surveys : pd.DataFrame
        Survey table with columns ``hole_id, depth, azimuth, dip``.
    assays : pd.DataFrame
        Assay table with columns ``hole_id, from_depth, to_depth`` plus
        any number of grade columns added via :meth:`add_assay`.

    Examples
    --------
    >>> db = DrillholeDB()
    >>> db.add_collar("DH001", 1000, 2000, 500, 100)
    >>> db.add_survey("DH001", 0, 0, -90)
    >>> db.add_assay("DH001", 0, 2, au_gpt=1.5)
    """

    def __init__(self) -> None:
        self.collars: pd.DataFrame = pd.DataFrame(columns=["hole_id", "x", "y", "z", "max_depth"])
        self.surveys: pd.DataFrame = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        self.assays: pd.DataFrame = pd.DataFrame(columns=["hole_id", "from_depth", "to_depth"])

    # ------------------------------------------------------------------
    # Data insertion
    # ------------------------------------------------------------------

    def add_collar(
        self,
        hole_id: str,
        x: float,
        y: float,
        z: float,
        max_depth: float,
    ) -> None:
        """Add a drillhole collar record.

        Parameters
        ----------
        hole_id : str
            Unique identifier for the drillhole.
        x, y, z : float
            Collar coordinates (easting, northing, elevation).
        max_depth : float
            Total planned / drilled depth of the hole.
        """
        row = pd.DataFrame([{"hole_id": hole_id, "x": x, "y": y, "z": z, "max_depth": max_depth}])
        self.collars = pd.concat([self.collars, row], ignore_index=True)

    def add_survey(
        self,
        hole_id: str,
        depth: float,
        azimuth: float,
        dip: float,
    ) -> None:
        """Add a downhole survey measurement.

        Parameters
        ----------
        hole_id : str
            Drillhole identifier (must exist in collars for validation to pass).
        depth : float
            Depth along hole at which measurement was taken.
        azimuth : float
            Bearing in degrees from north (0--360).
        dip : float
            Inclination from horizontal in degrees. Negative values point
            downward (mining convention: -90 = vertical down).
        """
        row = pd.DataFrame([{"hole_id": hole_id, "depth": depth, "azimuth": azimuth, "dip": dip}])
        self.surveys = pd.concat([self.surveys, row], ignore_index=True)

    def add_assay(
        self,
        hole_id: str,
        from_depth: float,
        to_depth: float,
        **grades: float,
    ) -> None:
        """Add an assay interval with one or more grade values.

        Parameters
        ----------
        hole_id : str
            Drillhole identifier.
        from_depth : float
            Start depth of the interval.
        to_depth : float
            End depth of the interval.
        **grades : float
            Keyword arguments for grade columns (e.g. ``au_gpt=1.5``).

        Raises
        ------
        ValueError
            If *from_depth* >= *to_depth*.
        """
        if from_depth >= to_depth:
            raise ValueError(f"from_depth ({from_depth}) must be less than to_depth ({to_depth})")

        record: dict = {
            "hole_id": hole_id,
            "from_depth": from_depth,
            "to_depth": to_depth,
        }
        record.update(grades)
        row = pd.DataFrame([record])
        self.assays = pd.concat([self.assays, row], ignore_index=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run basic integrity checks on the database.

        Returns
        -------
        list[str]
            A list of warning / error messages.  An empty list means no
            issues were found.

        Notes
        -----
        Checks performed:

        * Duplicate collar IDs.
        * Survey records referencing holes not in the collar table.
        * Assay records referencing holes not in the collar table.
        * Assay intervals that exceed *max_depth* recorded in the collar.
        * Survey depths that exceed *max_depth* recorded in the collar.
        """
        messages: list[str] = []

        # 1. Duplicate collars
        dup_collars = self.collars[self.collars["hole_id"].duplicated(keep=False)]
        if not dup_collars.empty:
            dup_ids = dup_collars["hole_id"].unique().tolist()
            messages.append(f"Duplicate collar IDs: {dup_ids}")

        collar_ids = set(self.collars["hole_id"].tolist())

        # 2. Orphan surveys
        if not self.surveys.empty:
            orphan_surv = set(self.surveys["hole_id"].tolist()) - collar_ids
            if orphan_surv:
                messages.append(f"Survey records with no collar: {sorted(orphan_surv)}")

        # 3. Orphan assays
        if not self.assays.empty:
            orphan_assay = set(self.assays["hole_id"].tolist()) - collar_ids
            if orphan_assay:
                messages.append(f"Assay records with no collar: {sorted(orphan_assay)}")

        # 4. Assay intervals exceeding max depth
        if not self.assays.empty and not self.collars.empty:
            depth_map = dict(zip(self.collars["hole_id"], self.collars["max_depth"], strict=False))
            for _, row in self.assays.iterrows():
                hid = row["hole_id"]
                if hid in depth_map:
                    md = float(depth_map[hid])
                    if float(row["to_depth"]) > md + 0.01:
                        messages.append(
                            f"Assay to_depth {row['to_depth']} exceeds max_depth "
                            f"{md} for hole {hid}"
                        )

        # 5. Survey depths exceeding max depth
        if not self.surveys.empty and not self.collars.empty:
            depth_map = dict(zip(self.collars["hole_id"], self.collars["max_depth"], strict=False))
            for _, row in self.surveys.iterrows():
                hid = row["hole_id"]
                if hid in depth_map:
                    md = float(depth_map[hid])
                    if float(row["depth"]) > md + 0.01:
                        messages.append(
                            f"Survey depth {row['depth']} exceeds max_depth {md} for hole {hid}"
                        )

        return messages

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Merge collar, survey, and assay tables into a single DataFrame.

        The merge is performed on ``hole_id``.  Survey rows are matched to
        assay intervals by finding the survey measurement whose depth is
        closest to the midpoint of the assay interval.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with collar coordinates, survey angles,
            and assay grades for each interval.
        """
        if self.assays.empty:
            return pd.DataFrame()

        # Merge collars into assays
        merged = self.assays.merge(self.collars, on="hole_id", how="left")

        if self.surveys.empty:
            return merged

        # For each assay interval, find the closest survey by depth
        survey_rows: list[pd.Series | None] = []
        for _, row in merged.iterrows():
            hid = row["hole_id"]
            mid = (float(row["from_depth"]) + float(row["to_depth"])) / 2.0
            hole_surveys = self.surveys[self.surveys["hole_id"] == hid].copy()
            if hole_surveys.empty:
                survey_rows.append(None)
                continue
            diffs = (hole_surveys["depth"].astype(float) - mid).abs()
            best_idx = diffs.idxmin()
            survey_rows.append(hole_surveys.loc[best_idx])

        survey_df = pd.DataFrame(survey_rows)
        if not survey_df.empty:
            survey_cols = ["azimuth", "dip"]
            existing = [c for c in survey_cols if c in survey_df.columns]
            if existing:
                # Reset indices to align
                for col in existing:
                    merged[col] = survey_df[col].values

        return merged

    def __repr__(self) -> str:
        return (
            f"DrillholeDB(collars={len(self.collars)}, "
            f"surveys={len(self.surveys)}, "
            f"assays={len(self.assays)})"
        )
