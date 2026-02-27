"""File I/O utilities for common mining data formats.

Supports reading and writing:

* **GSLIB / GeoEAS** -- the classic geostatistical data format.
* **CSV drillhole files** -- collar, survey, and assay as separate CSVs.
* **Block-model CSV** -- export of block model DataFrames.

References
----------
Deutsch, C. V. & Journel, A. G. (1998). *GSLIB: Geostatistical Software
Library and User's Guide*, 2nd ed.  Oxford University Press.  Appendix A.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pandas as pd

from minelab.data_management.drillholes import DrillholeDB


def read_gslib(filepath: str | Path) -> pd.DataFrame:
    """Read a GSLIB / GeoEAS formatted file into a DataFrame.

    Format
    ------
    ::

        Title line
        n_variables
        var1_name
        var2_name
        ...
        data rows (whitespace separated)

    Parameters
    ----------
    filepath : str or Path
        Path to the GSLIB file.

    Returns
    -------
    pd.DataFrame
        DataFrame with column names read from the header.

    Examples
    --------
    >>> df = read_gslib("data/samples.gslib")
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as fh:
        title = fh.readline().strip()  # noqa: F841 -- kept for potential future use
        n_vars = int(fh.readline().strip())
        col_names = [fh.readline().strip() for _ in range(n_vars)]

        data_lines: list[list[str]] = []
        for line in fh:
            stripped = line.strip()
            if stripped:
                data_lines.append(stripped.split())

    df = pd.DataFrame(data_lines, columns=col_names)

    # Convert numeric columns
    for col in df.columns:
        with contextlib.suppress(ValueError, TypeError):
            df[col] = pd.to_numeric(df[col])

    return df


def write_gslib(
    df: pd.DataFrame,
    filepath: str | Path,
    title: str = "GSLIB file",
) -> None:
    """Write a DataFrame to GSLIB / GeoEAS format.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    filepath : str or Path
        Output file path.
    title : str, optional
        Title line written at the top of the file (default ``"GSLIB file"``).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{len(df.columns)}\n")
        for col in df.columns:
            fh.write(f"{col}\n")
        for _, row in df.iterrows():
            values = [str(v) for v in row]
            fh.write(" ".join(values) + "\n")


def read_csv_drillholes(
    collar_path: str | Path,
    survey_path: str | Path,
    assay_path: str | Path,
) -> DrillholeDB:
    """Read standard CSV drillhole files and return a populated DrillholeDB.

    Expected CSV columns:

    * **Collar**: ``hole_id, x, y, z, max_depth``
    * **Survey**: ``hole_id, depth, azimuth, dip``
    * **Assay**: ``hole_id, from_depth, to_depth`` plus grade columns

    Parameters
    ----------
    collar_path, survey_path, assay_path : str or Path
        Paths to the respective CSV files.

    Returns
    -------
    DrillholeDB
        Populated drillhole database.
    """
    collar_df = pd.read_csv(collar_path)
    survey_df = pd.read_csv(survey_path)
    assay_df = pd.read_csv(assay_path)

    db = DrillholeDB()

    # Load collars
    for _, row in collar_df.iterrows():
        db.add_collar(
            hole_id=str(row["hole_id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            max_depth=float(row["max_depth"]),
        )

    # Load surveys
    for _, row in survey_df.iterrows():
        db.add_survey(
            hole_id=str(row["hole_id"]),
            depth=float(row["depth"]),
            azimuth=float(row["azimuth"]),
            dip=float(row["dip"]),
        )

    # Load assays -- grade columns are everything beyond the 3 required
    grade_cols = [c for c in assay_df.columns if c not in ("hole_id", "from_depth", "to_depth")]
    for _, row in assay_df.iterrows():
        grades = {c: float(row[c]) for c in grade_cols if pd.notna(row[c])}
        db.add_assay(
            hole_id=str(row["hole_id"]),
            from_depth=float(row["from_depth"]),
            to_depth=float(row["to_depth"]),
            **grades,
        )

    return db


def export_block_model_csv(
    blocks_df: pd.DataFrame,
    filepath: str | Path,
) -> None:
    """Export a block-model DataFrame to CSV.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Block model data.  Typically contains columns such as
        ``x, y, z, grade, tonnage, rock_type``.
    filepath : str or Path
        Destination CSV file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    blocks_df.to_csv(filepath, index=False)
