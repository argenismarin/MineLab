"""Tests for minelab.data_management.drillholes module."""

import pandas as pd
import pytest

from minelab.data_management.drillholes import DrillholeDB


class TestDrillholeDBInit:
    """Test DrillholeDB initialisation."""

    def test_empty_init(self):
        db = DrillholeDB()
        assert len(db.collars) == 0
        assert len(db.surveys) == 0
        assert len(db.assays) == 0

    def test_repr(self):
        db = DrillholeDB()
        assert "DrillholeDB" in repr(db)


class TestAddCollar:
    """Test collar insertion."""

    def test_add_single_collar(self):
        db = DrillholeDB()
        db.add_collar("DH001", 1000.0, 2000.0, 500.0, 100.0)
        assert len(db.collars) == 1
        assert db.collars.iloc[0]["hole_id"] == "DH001"
        assert float(db.collars.iloc[0]["x"]) == 1000.0

    def test_add_multiple_collars(self, sample_drillhole_collar):
        db = DrillholeDB()
        for _, row in sample_drillhole_collar.iterrows():
            db.add_collar(row["hole_id"], row["x"], row["y"], row["z"], row["max_depth"])
        assert len(db.collars) == 3


class TestAddSurvey:
    """Test survey insertion."""

    def test_add_survey(self):
        db = DrillholeDB()
        db.add_survey("DH001", 0.0, 0.0, -90.0)
        assert len(db.surveys) == 1
        assert float(db.surveys.iloc[0]["dip"]) == -90.0


class TestAddAssay:
    """Test assay insertion."""

    def test_add_assay_with_grade(self):
        db = DrillholeDB()
        db.add_assay("DH001", 0.0, 2.0, au_gpt=1.5, cu_pct=0.3)
        assert len(db.assays) == 1
        assert float(db.assays.iloc[0]["au_gpt"]) == 1.5
        assert float(db.assays.iloc[0]["cu_pct"]) == 0.3

    def test_add_assay_invalid_interval(self):
        db = DrillholeDB()
        with pytest.raises(ValueError, match="from_depth"):
            db.add_assay("DH001", 5.0, 3.0, au_gpt=1.0)


class TestValidate:
    """Test database validation."""

    def test_valid_database(self, sample_drillhole_collar, sample_drillhole_survey,
                            sample_drillhole_assay):
        db = DrillholeDB()
        for _, row in sample_drillhole_collar.iterrows():
            db.add_collar(row["hole_id"], row["x"], row["y"], row["z"], row["max_depth"])
        for _, row in sample_drillhole_survey.iterrows():
            db.add_survey(row["hole_id"], row["depth"], row["azimuth"], row["dip"])
        for _, row in sample_drillhole_assay.iterrows():
            grades = {c: float(row[c]) for c in ["au_gpt"]}
            db.add_assay(row["hole_id"], float(row["from_depth"]),
                         float(row["to_depth"]), **grades)
        messages = db.validate()
        assert messages == []

    def test_duplicate_collars(self):
        db = DrillholeDB()
        db.add_collar("DH001", 100, 200, 300, 50)
        db.add_collar("DH001", 101, 201, 301, 51)
        messages = db.validate()
        assert any("Duplicate" in m for m in messages)

    def test_orphan_survey(self):
        db = DrillholeDB()
        db.add_collar("DH001", 100, 200, 300, 50)
        db.add_survey("DH999", 0, 0, -90)  # no collar for DH999
        messages = db.validate()
        assert any("no collar" in m for m in messages)

    def test_orphan_assay(self):
        db = DrillholeDB()
        db.add_collar("DH001", 100, 200, 300, 50)
        db.add_assay("DH999", 0, 2, au=1.0)
        messages = db.validate()
        assert any("no collar" in m for m in messages)

    def test_assay_exceeds_max_depth(self):
        db = DrillholeDB()
        db.add_collar("DH001", 100, 200, 300, 50)
        db.add_assay("DH001", 49, 55, au=1.0)
        messages = db.validate()
        assert any("exceeds max_depth" in m for m in messages)

    def test_survey_exceeds_max_depth(self):
        db = DrillholeDB()
        db.add_collar("DH001", 100, 200, 300, 50)
        db.add_survey("DH001", 60, 0, -90)
        messages = db.validate()
        assert any("exceeds max_depth" in m for m in messages)


class TestToDataFrame:
    """Test merged output."""

    def test_to_dataframe_with_fixtures(self, sample_drillhole_collar,
                                        sample_drillhole_survey,
                                        sample_drillhole_assay):
        db = DrillholeDB()
        for _, row in sample_drillhole_collar.iterrows():
            db.add_collar(row["hole_id"], row["x"], row["y"], row["z"], row["max_depth"])
        for _, row in sample_drillhole_survey.iterrows():
            db.add_survey(row["hole_id"], row["depth"], row["azimuth"], row["dip"])
        for _, row in sample_drillhole_assay.iterrows():
            grades = {c: float(row[c]) for c in ["au_gpt"]}
            db.add_assay(row["hole_id"], float(row["from_depth"]),
                         float(row["to_depth"]), **grades)

        merged = db.to_dataframe()
        assert not merged.empty
        # Must contain collar columns
        assert "x" in merged.columns
        assert "y" in merged.columns
        assert "z" in merged.columns
        # Must contain grade columns
        assert "au_gpt" in merged.columns
        # Must contain survey columns
        assert "azimuth" in merged.columns
        assert "dip" in merged.columns
        # Row count should match assay count
        assert len(merged) == len(sample_drillhole_assay)

    def test_to_dataframe_empty(self):
        db = DrillholeDB()
        assert db.to_dataframe().empty
