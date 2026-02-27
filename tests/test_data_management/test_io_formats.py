"""Tests for minelab.data_management.io_formats module."""

import numpy as np
import pandas as pd
import pytest

from minelab.data_management.io_formats import (
    export_block_model_csv,
    read_csv_drillholes,
    read_gslib,
    write_gslib,
)


@pytest.fixture
def sample_df():
    """Small DataFrame for I/O testing."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0],
        "grade": [0.5, 1.2, 0.8],
    })


class TestGSLIBRoundTrip:
    """Test GSLIB write then read produces identical data."""

    def test_roundtrip_numeric(self, sample_df, tmp_path):
        filepath = tmp_path / "test.gslib"
        write_gslib(sample_df, filepath, title="Test data")
        result = read_gslib(filepath)

        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)
        np.testing.assert_allclose(result["x"].values, sample_df["x"].values)
        np.testing.assert_allclose(result["y"].values, sample_df["y"].values)
        np.testing.assert_allclose(result["grade"].values, sample_df["grade"].values)

    def test_roundtrip_preserves_columns(self, tmp_path):
        df = pd.DataFrame({
            "easting": [100.0],
            "northing": [200.0],
            "elevation": [50.0],
            "au_gpt": [3.5],
            "cu_pct": [0.7],
        })
        filepath = tmp_path / "multi.gslib"
        write_gslib(df, filepath)
        result = read_gslib(filepath)
        assert list(result.columns) == ["easting", "northing", "elevation", "au_gpt", "cu_pct"]

    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame({"a": [], "b": []})
        filepath = tmp_path / "empty.gslib"
        write_gslib(df, filepath)
        result = read_gslib(filepath)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 0

    def test_title_preserved_in_file(self, sample_df, tmp_path):
        filepath = tmp_path / "titled.gslib"
        write_gslib(sample_df, filepath, title="My Custom Title")
        with open(filepath) as fh:
            first_line = fh.readline().strip()
        assert first_line == "My Custom Title"

    def test_large_roundtrip(self, tmp_path):
        """Round-trip a larger dataset."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.uniform(0, 1000, 500),
            "y": rng.uniform(0, 1000, 500),
            "z": rng.uniform(0, 500, 500),
            "grade": rng.lognormal(0, 1, 500),
        })
        filepath = tmp_path / "large.gslib"
        write_gslib(df, filepath)
        result = read_gslib(filepath)
        assert len(result) == 500
        np.testing.assert_allclose(result["x"].values, df["x"].values, rtol=1e-6)


class TestCSVDrillholes:
    """Test reading CSV drillhole files."""

    def test_read_csv_drillholes(self, tmp_path, sample_drillhole_collar,
                                  sample_drillhole_survey, sample_drillhole_assay):
        collar_path = tmp_path / "collars.csv"
        survey_path = tmp_path / "surveys.csv"
        assay_path = tmp_path / "assays.csv"

        sample_drillhole_collar.to_csv(collar_path, index=False)
        sample_drillhole_survey.to_csv(survey_path, index=False)
        sample_drillhole_assay.to_csv(assay_path, index=False)

        db = read_csv_drillholes(collar_path, survey_path, assay_path)
        assert len(db.collars) == 3
        assert len(db.surveys) == len(sample_drillhole_survey)
        assert len(db.assays) == len(sample_drillhole_assay)
        # Validate should pass
        messages = db.validate()
        assert messages == []


class TestExportBlockModel:
    """Test block model CSV export."""

    def test_export_and_readback(self, tmp_path):
        blocks = pd.DataFrame({
            "x": [10, 20, 30],
            "y": [10, 20, 30],
            "z": [5, 5, 5],
            "grade": [1.0, 2.0, 3.0],
            "tonnage": [1000, 1000, 1000],
        })
        filepath = tmp_path / "blocks.csv"
        export_block_model_csv(blocks, filepath)
        result = pd.read_csv(filepath)
        assert len(result) == 3
        assert list(result.columns) == ["x", "y", "z", "grade", "tonnage"]
        np.testing.assert_allclose(result["grade"].values, [1.0, 2.0, 3.0])

    def test_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "subdir" / "deep" / "blocks.csv"
        blocks = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        export_block_model_csv(blocks, filepath)
        assert filepath.exists()
