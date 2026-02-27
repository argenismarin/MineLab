"""Tests for minelab.data_management.compositing module."""

import numpy as np
import pandas as pd
import pytest

from minelab.data_management.compositing import (
    composite_by_bench,
    composite_by_geology,
    composite_by_length,
)


@pytest.fixture
def simple_assays():
    """5 x 2m samples for a single hole, grades 1..5."""
    return pd.DataFrame({
        "hole_id": ["DH1"] * 5,
        "from_depth": [0.0, 2.0, 4.0, 6.0, 8.0],
        "to_depth": [2.0, 4.0, 6.0, 8.0, 10.0],
        "au": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def two_hole_assays():
    """Two holes, 2m samples."""
    rows = []
    for hid, base_grade in [("DH1", 1.0), ("DH2", 10.0)]:
        for i in range(5):
            rows.append({
                "hole_id": hid,
                "from_depth": float(i * 2),
                "to_depth": float((i + 1) * 2),
                "au": base_grade + i,
            })
    return pd.DataFrame(rows)


class TestCompositeByLength:
    """Length compositing tests."""

    def test_identity_composite(self, simple_assays):
        """Composite length == sample length should return same grades."""
        comp = composite_by_length(simple_assays, 2.0)
        assert len(comp) == 5
        np.testing.assert_allclose(comp["au"].values, [1, 2, 3, 4, 5])

    def test_full_hole_composite(self, simple_assays):
        """Composite length == hole length: length-weighted average."""
        comp = composite_by_length(simple_assays, 10.0)
        assert len(comp) == 1
        # (1*2+2*2+3*2+4*2+5*2) / 10 = 30/10 = 3.0
        np.testing.assert_allclose(comp["au"].iloc[0], 3.0)

    def test_two_composites_from_five(self, simple_assays):
        """5 x 2m samples -> 2 x 5m composites, verify length-weighting."""
        comp = composite_by_length(simple_assays, 5.0)
        assert len(comp) == 2
        # First composite [0-5): samples 0-2(full), 2-4(full), 4-5(half of 4-6)
        # Wait, samples are [0,2), [2,4), [4,6)... composite [0,5) overlaps:
        #   [0,2) full -> weight=2, grade=1
        #   [2,4) full -> weight=2, grade=2
        #   [4,5) partial-> weight=1, grade=3
        # avg = (1*2 + 2*2 + 3*1) / 5 = 9/5 = 1.8
        np.testing.assert_allclose(comp["au"].iloc[0], 1.8, atol=1e-10)
        # Second composite [5-10): samples
        #   [4,6) partial [5,6) -> weight=1, grade=3
        #   [6,8) full -> weight=2, grade=4
        #   [8,10) full -> weight=2, grade=5
        # avg = (3*1 + 4*2 + 5*2) / 5 = 21/5 = 4.2
        np.testing.assert_allclose(comp["au"].iloc[1], 4.2, atol=1e-10)

    def test_multiple_holes(self, two_hole_assays):
        """Each hole composited independently."""
        comp = composite_by_length(two_hole_assays, 10.0)
        assert len(comp) == 2
        dh1 = comp[comp["hole_id"] == "DH1"]
        dh2 = comp[comp["hole_id"] == "DH2"]
        np.testing.assert_allclose(dh1["au"].iloc[0], 3.0)
        np.testing.assert_allclose(dh2["au"].iloc[0], 12.0)

    def test_length_weighted_unequal_samples(self):
        """Verify length-weighting with unequal sample lengths."""
        assays = pd.DataFrame({
            "hole_id": ["DH1", "DH1"],
            "from_depth": [0.0, 1.0],
            "to_depth": [1.0, 5.0],
            "au": [10.0, 2.0],
        })
        comp = composite_by_length(assays, 10.0)
        # (10*1 + 2*4) / 5 = 18/5 = 3.6
        np.testing.assert_allclose(comp["au"].iloc[0], 3.6, atol=1e-10)

    def test_fixture_2m_to_10m(self, sample_drillhole_assay):
        """Composite the conftest 2m assays into 10m composites."""
        comp = composite_by_length(sample_drillhole_assay, 10.0, grade_cols=["au_gpt"])
        # DH001 has 50 x 2m samples (0-100m) -> 10 x 10m composites
        dh1 = comp[comp["hole_id"] == "DH001"]
        assert len(dh1) == 10
        # Each 10m composite should be average of 5 x 2m samples
        assert all(dh1["au_gpt"].notna())


class TestCompositeByGeology:
    """Geology-unit compositing tests."""

    def test_single_unit(self):
        assays = pd.DataFrame({
            "hole_id": ["DH1"] * 4,
            "from_depth": [0, 2, 4, 6],
            "to_depth": [2, 4, 6, 8],
            "au": [1.0, 2.0, 3.0, 4.0],
            "lith": ["OX", "OX", "OX", "OX"],
        })
        comp = composite_by_geology(assays, "lith")
        assert len(comp) == 1
        np.testing.assert_allclose(comp["au"].iloc[0], 2.5)

    def test_two_units(self):
        assays = pd.DataFrame({
            "hole_id": ["DH1"] * 4,
            "from_depth": [0, 2, 4, 6],
            "to_depth": [2, 4, 6, 8],
            "au": [1.0, 2.0, 3.0, 4.0],
            "lith": ["OX", "OX", "SUL", "SUL"],
        })
        comp = composite_by_geology(assays, "lith")
        assert len(comp) == 2
        ox = comp[comp["lith"] == "OX"]
        sul = comp[comp["lith"] == "SUL"]
        np.testing.assert_allclose(ox["au"].iloc[0], 1.5)
        np.testing.assert_allclose(sul["au"].iloc[0], 3.5)

    def test_unequal_lengths(self):
        """Length-weighting within geology units."""
        assays = pd.DataFrame({
            "hole_id": ["DH1", "DH1"],
            "from_depth": [0.0, 1.0],
            "to_depth": [1.0, 5.0],
            "au": [10.0, 2.0],
            "lith": ["A", "A"],
        })
        comp = composite_by_geology(assays, "lith")
        # (10*1 + 2*4) / 5 = 3.6
        np.testing.assert_allclose(comp["au"].iloc[0], 3.6, atol=1e-10)


class TestCompositeByBench:
    """Bench compositing tests."""

    def test_simple_bench(self):
        """Vertical hole at z=100, 10m bench."""
        assays = pd.DataFrame({
            "hole_id": ["DH1"] * 5,
            "from_depth": [0, 2, 4, 6, 8],
            "to_depth": [2, 4, 6, 8, 10],
            "au": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [100.0] * 5,
        })
        comp = composite_by_bench(assays, 10.0, collar_z_col="z")
        # All samples fall within elevation 90-100, which is one 10m bench
        assert len(comp) == 1
        np.testing.assert_allclose(comp["au"].iloc[0], 3.0)

    def test_two_benches(self):
        """Samples spanning two 5m benches."""
        assays = pd.DataFrame({
            "hole_id": ["DH1"] * 5,
            "from_depth": [0, 2, 4, 6, 8],
            "to_depth": [2, 4, 6, 8, 10],
            "au": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [100.0] * 5,
        })
        comp = composite_by_bench(assays, 5.0, collar_z_col="z")
        # Bench [95-100] captures samples at depth [0-5m] elev [95-100]
        # Bench [90-95] captures samples at depth [5-10m] elev [90-95]
        assert len(comp) == 2
