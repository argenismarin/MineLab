"""Integration test for round-trip I/O of drillhole and geological data.

Tests that data written to disk and read back maintains full integrity,
and that compositing preserves grade-length consistency across the
save/load cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minelab.data_management import (
    DrillholeDB,
    composite_by_length,
    read_csv_drillholes,
    read_gslib,
    write_gslib,
)


def _sample_drillhole_db() -> DrillholeDB:
    """Build a small drillhole database for I/O testing."""
    db = DrillholeDB()
    collars = [
        ("DH001", 100.0, 200.0, 500.0, 50.0),
        ("DH002", 150.0, 200.0, 510.0, 40.0),
        ("DH003", 200.0, 250.0, 505.0, 45.0),
    ]
    for hid, x, y, z, md in collars:
        db.add_collar(hid, x, y, z, md)
        db.add_survey(hid, 0.0, 0.0, -90.0)
        db.add_survey(hid, md / 2.0, 5.0, -88.0)

    rng = np.random.default_rng(123)
    for hid, _, _, _, md in collars:
        depth = 0.0
        while depth < md:
            interval = min(2.0, md - depth)
            au = max(0.0, round(rng.lognormal(0.0, 0.8), 3))
            cu = max(0.0, round(rng.normal(0.5, 0.2), 3))
            db.add_assay(hid, depth, depth + interval, au=au, cu=cu)
            depth += interval

    return db


@pytest.fixture()
def sample_db() -> DrillholeDB:
    """Fixture providing a small drillhole database."""
    return _sample_drillhole_db()


class TestCSVRoundTrip:
    """Write collar/survey/assay CSVs and read them back."""

    def test_csv_roundtrip_collars(self, sample_db: DrillholeDB, tmp_path):
        """Collar data should survive a CSV round-trip."""
        collar_path = tmp_path / "collar.csv"
        survey_path = tmp_path / "survey.csv"
        assay_path = tmp_path / "assay.csv"

        sample_db.collars.to_csv(collar_path, index=False)
        sample_db.surveys.to_csv(survey_path, index=False)
        sample_db.assays.to_csv(assay_path, index=False)

        reloaded = read_csv_drillholes(collar_path, survey_path, assay_path)

        assert len(reloaded.collars) == len(sample_db.collars)
        for col in ["hole_id", "x", "y", "z", "max_depth"]:
            original = sample_db.collars[col].tolist()
            loaded = reloaded.collars[col].tolist()
            if col == "hole_id":
                assert original == loaded
            else:
                for a, b in zip(original, loaded):
                    assert float(a) == pytest.approx(float(b), abs=1e-6)

    def test_csv_roundtrip_assays(self, sample_db: DrillholeDB, tmp_path):
        """Assay grades should survive a CSV round-trip."""
        collar_path = tmp_path / "collar.csv"
        survey_path = tmp_path / "survey.csv"
        assay_path = tmp_path / "assay.csv"

        sample_db.collars.to_csv(collar_path, index=False)
        sample_db.surveys.to_csv(survey_path, index=False)
        sample_db.assays.to_csv(assay_path, index=False)

        reloaded = read_csv_drillholes(collar_path, survey_path, assay_path)

        assert len(reloaded.assays) == len(sample_db.assays)
        orig_au = sample_db.assays["au"].astype(float).tolist()
        load_au = reloaded.assays["au"].astype(float).tolist()
        for a, b in zip(orig_au, load_au):
            assert a == pytest.approx(b, abs=1e-6)

    def test_csv_roundtrip_surveys(self, sample_db: DrillholeDB, tmp_path):
        """Survey data should survive a CSV round-trip."""
        collar_path = tmp_path / "collar.csv"
        survey_path = tmp_path / "survey.csv"
        assay_path = tmp_path / "assay.csv"

        sample_db.collars.to_csv(collar_path, index=False)
        sample_db.surveys.to_csv(survey_path, index=False)
        sample_db.assays.to_csv(assay_path, index=False)

        reloaded = read_csv_drillholes(collar_path, survey_path, assay_path)

        assert len(reloaded.surveys) == len(sample_db.surveys)
        orig_dip = sample_db.surveys["dip"].astype(float).tolist()
        load_dip = reloaded.surveys["dip"].astype(float).tolist()
        for a, b in zip(orig_dip, load_dip):
            assert a == pytest.approx(b, abs=1e-6)


class TestGSLIBRoundTrip:
    """Write a DataFrame to GSLIB format and read it back."""

    def test_gslib_roundtrip_numeric(self, tmp_path):
        """Numeric data should survive GSLIB write/read cycle."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "grade": [0.5, 1.2, 0.8, 2.1],
        })
        filepath = tmp_path / "test.gslib"
        write_gslib(df, filepath, title="Test data")
        reloaded = read_gslib(filepath)

        assert list(reloaded.columns) == ["x", "y", "grade"]
        assert len(reloaded) == 4
        for col in df.columns:
            for orig, loaded in zip(df[col], reloaded[col]):
                assert float(orig) == pytest.approx(float(loaded), abs=1e-6)

    def test_gslib_preserves_column_names(self, tmp_path):
        """Column names should be preserved through GSLIB write/read."""
        df = pd.DataFrame({
            "easting": [100.0],
            "northing": [200.0],
            "elevation": [50.0],
            "au_ppm": [3.5],
        })
        filepath = tmp_path / "cols.gslib"
        write_gslib(df, filepath)
        reloaded = read_gslib(filepath)

        assert list(reloaded.columns) == ["easting", "northing", "elevation", "au_ppm"]


class TestCompositingRoundTrip:
    """Composite, save, reload, and verify grade-length consistency."""

    def test_composite_save_reload(self, sample_db: DrillholeDB, tmp_path):
        """Composited data should survive a CSV round-trip."""
        composites = composite_by_length(
            sample_db.assays, length=5.0, grade_cols=["au", "cu"]
        )
        csv_path = tmp_path / "composites.csv"
        composites.to_csv(csv_path, index=False)
        reloaded = pd.read_csv(csv_path)

        assert len(reloaded) == len(composites)
        for col in ["au", "cu"]:
            orig = composites[col].astype(float).values
            load = reloaded[col].astype(float).values
            np.testing.assert_allclose(orig, load, atol=1e-6)

    def test_composite_grades_are_length_weighted(
        self, sample_db: DrillholeDB
    ):
        """Compositing should produce length-weighted averages that preserve
        the overall mean grade when composite length divides evenly."""
        assays = sample_db.assays
        composites = composite_by_length(assays, length=10.0, grade_cols=["au"])

        # For each hole, the length-weighted mean of originals should match
        # the length-weighted mean of composites
        for hid in assays["hole_id"].unique():
            orig = assays[assays["hole_id"] == hid]
            comp = composites[composites["hole_id"] == hid]

            orig_lengths = (
                orig["to_depth"].astype(float) - orig["from_depth"].astype(float)
            )
            comp_lengths = (
                comp["to_depth"].astype(float) - comp["from_depth"].astype(float)
            )

            orig_mean = np.average(
                orig["au"].astype(float), weights=orig_lengths
            )
            comp_mean = np.average(
                comp["au"].astype(float), weights=comp_lengths
            )
            assert orig_mean == pytest.approx(comp_mean, rel=1e-4)

    def test_validation_passes_after_reload(
        self, sample_db: DrillholeDB, tmp_path
    ):
        """A valid database should produce no validation errors after reload."""
        collar_path = tmp_path / "collar.csv"
        survey_path = tmp_path / "survey.csv"
        assay_path = tmp_path / "assay.csv"

        sample_db.collars.to_csv(collar_path, index=False)
        sample_db.surveys.to_csv(survey_path, index=False)
        sample_db.assays.to_csv(assay_path, index=False)

        reloaded = read_csv_drillholes(collar_path, survey_path, assay_path)
        errors = reloaded.validate()
        assert errors == []
