"""Tests for minelab.data_management.validation module."""

import pandas as pd
import pytest

from minelab.data_management.drillholes import DrillholeDB
from minelab.data_management.validation import (
    check_assay_gaps,
    check_assay_overlaps,
    check_collar_duplicates,
    check_survey_consistency,
    validation_report,
)


class TestCheckCollarDuplicates:
    """Test collar duplicate detection."""

    def test_no_duplicates(self):
        df = pd.DataFrame({"hole_id": ["A", "B", "C"]})
        assert check_collar_duplicates(df) == []

    def test_single_duplicate(self):
        df = pd.DataFrame({"hole_id": ["A", "B", "A"]})
        result = check_collar_duplicates(df)
        assert result == ["A"]

    def test_multiple_duplicates(self):
        df = pd.DataFrame({"hole_id": ["A", "B", "A", "B", "C"]})
        result = check_collar_duplicates(df)
        assert sorted(result) == ["A", "B"]

    def test_all_same(self):
        df = pd.DataFrame({"hole_id": ["X", "X", "X"]})
        result = check_collar_duplicates(df)
        assert result == ["X"]


class TestCheckSurveyConsistency:
    """Test survey range validation."""

    def test_valid_surveys(self):
        df = pd.DataFrame({
            "hole_id": ["DH1", "DH1"],
            "depth": [0.0, 50.0],
            "azimuth": [45.0, 90.0],
            "dip": [-90.0, -60.0],
        })
        assert check_survey_consistency(df) == []

    def test_dip_out_of_range(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"],
            "depth": [0.0],
            "azimuth": [0.0],
            "dip": [-95.0],
        })
        issues = check_survey_consistency(df)
        assert len(issues) == 1
        assert "dip" in issues[0]

    def test_azimuth_negative(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"],
            "depth": [0.0],
            "azimuth": [-10.0],
            "dip": [-45.0],
        })
        issues = check_survey_consistency(df)
        assert len(issues) == 1
        assert "azimuth" in issues[0]

    def test_azimuth_360(self):
        """Azimuth of exactly 360 is out of range [0, 360)."""
        df = pd.DataFrame({
            "hole_id": ["DH1"],
            "depth": [0.0],
            "azimuth": [360.0],
            "dip": [-45.0],
        })
        issues = check_survey_consistency(df)
        assert len(issues) == 1

    def test_both_invalid(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"],
            "depth": [0.0],
            "azimuth": [-5.0],
            "dip": [100.0],
        })
        issues = check_survey_consistency(df)
        assert len(issues) == 2


class TestCheckAssayOverlaps:
    """Test overlap detection."""

    def test_no_overlaps(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 3,
            "from_depth": [0, 2, 4],
            "to_depth": [2, 4, 6],
        })
        result = check_assay_overlaps(df)
        assert result.empty

    def test_single_overlap(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 3,
            "from_depth": [0.0, 1.5, 4.0],
            "to_depth": [2.0, 4.0, 6.0],
        })
        result = check_assay_overlaps(df)
        assert len(result) == 1
        assert float(result.iloc[0]["overlap"]) == pytest.approx(0.5)

    def test_multiple_overlaps(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 3,
            "from_depth": [0.0, 1.0, 2.0],
            "to_depth": [2.0, 3.0, 4.0],
        })
        result = check_assay_overlaps(df)
        assert len(result) == 2

    def test_overlaps_per_hole(self):
        """Overlaps checked independently per hole."""
        df = pd.DataFrame({
            "hole_id": ["DH1", "DH1", "DH2", "DH2"],
            "from_depth": [0, 1, 0, 2],
            "to_depth": [2, 3, 2, 4],
        })
        result = check_assay_overlaps(df)
        # DH1 has overlap, DH2 does not
        assert len(result) == 1
        assert result.iloc[0]["hole_id"] == "DH1"


class TestCheckAssayGaps:
    """Test gap detection."""

    def test_no_gaps(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 3,
            "from_depth": [0, 2, 4],
            "to_depth": [2, 4, 6],
        })
        result = check_assay_gaps(df)
        assert result.empty

    def test_single_gap(self):
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 3,
            "from_depth": [0.0, 3.0, 6.0],
            "to_depth": [2.0, 5.0, 8.0],
        })
        result = check_assay_gaps(df)
        assert len(result) == 2  # gap between [2,3) and [5,6)
        assert float(result.iloc[0]["gap"]) == pytest.approx(1.0)

    def test_tolerance(self):
        """Gap smaller than tolerance should not be reported."""
        df = pd.DataFrame({
            "hole_id": ["DH1"] * 2,
            "from_depth": [0.0, 2.005],
            "to_depth": [2.0, 4.0],
        })
        # Default tolerance 0.01 -> gap of 0.005 is OK
        result = check_assay_gaps(df, tolerance=0.01)
        assert result.empty

        # With tighter tolerance -> gap is flagged
        result_tight = check_assay_gaps(df, tolerance=0.001)
        assert len(result_tight) == 1


class TestValidationReport:
    """Test the full validation report."""

    def test_clean_database(self, sample_drillhole_collar, sample_drillhole_survey,
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

        report = validation_report(db)
        assert report["is_valid"] is True
        assert report["collar_duplicates"] == []
        assert report["survey_issues"] == []
        assert report["assay_overlaps"].empty
        assert report["assay_gaps"].empty

    def test_dirty_database(self):
        db = DrillholeDB()
        db.add_collar("DH1", 0, 0, 0, 50)
        db.add_collar("DH1", 1, 1, 1, 50)  # duplicate
        db.add_survey("DH1", 0, 0, -90)
        db.add_survey("DH1", 20, -5, -100)  # bad azimuth and dip
        db.add_assay("DH1", 0, 2, au=1.0)
        db.add_assay("DH1", 5, 8, au=2.0)  # gap between 2 and 5

        report = validation_report(db)
        assert report["is_valid"] is False
        assert "DH1" in report["collar_duplicates"]
        assert len(report["survey_issues"]) > 0
        assert not report["assay_gaps"].empty
