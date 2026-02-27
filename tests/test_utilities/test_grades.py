"""Tests for minelab.utilities.grades."""

import numpy as np
import pandas as pd
import pytest

from minelab.utilities.grades import (
    equivalent_grade,
    gpt_to_oz_per_ton,
    gpt_to_ppm,
    grade_tonnage_curve,
    metal_content,
    oz_per_ton_to_gpt,
    percent_to_ppm,
    ppm_to_gpt,
    ppm_to_percent,
)


class TestPpmPercent:
    def test_ppm_to_percent(self):
        assert ppm_to_percent(10_000) == 1.0

    def test_percent_to_ppm(self):
        assert percent_to_ppm(1.0) == 10_000.0

    def test_round_trip(self):
        val = 4567.0
        assert pytest.approx(percent_to_ppm(ppm_to_percent(val))) == val


class TestPpmGpt:
    def test_ppm_to_gpt_identity(self):
        assert ppm_to_gpt(5.0) == 5.0

    def test_gpt_to_ppm_identity(self):
        assert gpt_to_ppm(5.0) == 5.0


class TestOzTonGpt:
    """1 troy oz/short ton = 34.2857 g/t (literature value)."""

    def test_oz_per_ton_to_gpt(self):
        result = oz_per_ton_to_gpt(1.0)
        assert pytest.approx(result, abs=0.001) == 34.2857

    def test_gpt_to_oz_per_ton(self):
        result = gpt_to_oz_per_ton(34.2857)
        assert pytest.approx(result, abs=0.001) == 1.0

    def test_round_trip(self):
        val = 2.5
        converted = gpt_to_oz_per_ton(oz_per_ton_to_gpt(val))
        assert pytest.approx(converted, rel=1e-9) == val


class TestGradeTonnageCurve:
    def test_basic_curve(self):
        grades = [0.5, 1.0, 1.5, 2.0, 2.5]
        tonnages = [100, 100, 100, 100, 100]
        cutoffs = [0.0, 1.0, 2.0, 3.0]
        df = grade_tonnage_curve(grades, tonnages, cutoffs)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "cutoff", "tonnes_above", "mean_grade_above", "metal_above",
        ]

        # At cutoff 0.0, all blocks are included
        row0 = df.iloc[0]
        assert row0["tonnes_above"] == 500.0
        assert pytest.approx(row0["mean_grade_above"], rel=1e-6) == 1.5

        # At cutoff 2.0, only grades >= 2.0
        row2 = df.iloc[2]
        assert row2["tonnes_above"] == 200.0
        assert pytest.approx(row2["mean_grade_above"], rel=1e-6) == 2.25

        # At cutoff 3.0, no blocks
        row3 = df.iloc[3]
        assert row3["tonnes_above"] == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            grade_tonnage_curve([1, 2], [100], [0])

    def test_weighted_mean(self):
        """Variable tonnage blocks."""
        grades = [1.0, 2.0]
        tonnages = [200, 800]
        cutoffs = [0.0]
        df = grade_tonnage_curve(grades, tonnages, cutoffs)
        # Weighted mean: (1*200 + 2*800) / 1000 = 1.8
        assert pytest.approx(df.iloc[0]["mean_grade_above"], rel=1e-9) == 1.8


class TestMetalContent:
    def test_basic(self):
        assert metal_content(1_000_000, 0.005) == 5000.0

    def test_with_recovery(self):
        assert metal_content(1_000_000, 0.005, recovery=0.90) == 4500.0

    def test_zero_tonnage(self):
        assert metal_content(0, 0.01) == 0.0

    def test_negative_tonnage_raises(self):
        with pytest.raises(ValueError):
            metal_content(-100, 0.01)


class TestEquivalentGrade:
    def test_two_elements(self):
        # Cu-Au: Cu grade 1%, Au 0.5 g/t
        # Cu price $5000/t, Au $60/g; Recovery Cu 90%, Au 85%
        eq = equivalent_grade(
            grades=[1.0, 0.5],
            prices=[5000, 60],
            recoveries=[0.90, 0.85],
        )
        # eq = 1.0 + 0.5 * 60 * 0.85 / (5000 * 0.90)
        #    = 1.0 + 25.5 / 4500 = 1.005667
        expected = 1.0 + (0.5 * 60.0 * 0.85) / (5000.0 * 0.90)
        assert pytest.approx(eq, rel=1e-6) == expected

    def test_default_recoveries(self):
        eq = equivalent_grade([1.0, 2.0], [10, 5])
        # eq = 1.0 + 2.0 * 5 / 10 = 2.0
        assert pytest.approx(eq, rel=1e-9) == 2.0

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            equivalent_grade([1.0], [5000])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            equivalent_grade([1.0, 2.0], [5000, 60], [0.9])
