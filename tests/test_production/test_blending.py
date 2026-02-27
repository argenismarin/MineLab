"""Tests for minelab.production.blending."""

import pytest

from minelab.production.blending import (
    blend_grade,
    blend_optimize,
)


class TestBlendOptimize:
    """Tests for LP blending optimization."""

    def test_feasible(self):
        """Should find feasible blend with loose constraints."""
        sources = [
            {"tonnage_available": 1000, "grades": {"Cu": 2.0}},
            {"tonnage_available": 1000, "grades": {"Cu": 0.5}},
        ]
        constraints = {"Cu": {"min": 1.0, "max": 1.5}}
        result = blend_optimize(sources, constraints, 500)
        assert result["feasible"]

    def test_tonnage_target(self):
        """Total tonnage should meet target."""
        sources = [
            {"tonnage_available": 1000, "grades": {"Cu": 2.0}},
            {"tonnage_available": 1000, "grades": {"Cu": 0.5}},
        ]
        constraints = {"Cu": {"min": 1.0, "max": 1.5}}
        result = blend_optimize(sources, constraints, 500)
        if result["feasible"]:
            total = sum(result["tonnages"])
            assert total == pytest.approx(500, rel=0.05)


class TestBlendGrade:
    """Tests for weighted average grade."""

    def test_known_value(self):
        """Weighted average: (100*2 + 200*1) / 300 = 1.333."""
        g = blend_grade([100, 200], [2.0, 1.0])
        assert g == pytest.approx(1.333, rel=0.01)

    def test_equal_weights(self):
        """Equal tonnages → simple average."""
        g = blend_grade([100, 100], [2.0, 4.0])
        assert g == pytest.approx(3.0, rel=0.01)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestBlendGradeValidation:
    """Validation tests for blend_grade."""

    def test_empty_raises(self):
        """Empty source_tonnages should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            blend_grade([], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            blend_grade([100, 200], [1.5])

    def test_negative_tonnage_raises(self):
        """Negative tonnage should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            blend_grade([-100, 200], [1.5, 0.8])

    def test_zero_total_tonnage_raises(self):
        """Zero total tonnage should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            blend_grade([0, 0], [1.5, 0.8])


class TestBlendOptimizeValidation:
    """Validation tests for blend_optimize."""

    def test_empty_sources_raises(self):
        """Empty sources should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            blend_optimize([], {"Cu": {"min": 0.5}}, 500)


class TestBlendOptimizeInfeasible:
    """Test infeasible blend scenarios."""

    def test_infeasible_blend(self):
        """Impossible constraints should return infeasible result."""
        sources = [
            {"tonnage_available": 100, "grades": {"Cu": 0.1}},
            {"tonnage_available": 100, "grades": {"Cu": 0.2}},
        ]
        # Require Cu >= 5.0 which is impossible with sources at 0.1 and 0.2
        constraints = {"Cu": {"min": 5.0}}
        result = blend_optimize(sources, constraints, 150)
        assert result["feasible"] is False
        assert result["blend_grade"]["Cu"] == 0.0
