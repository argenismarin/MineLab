"""Tests for minelab.drilling_blasting.blast_design."""

import pytest

from minelab.drilling_blasting.blast_design import (
    burden_konya,
    burden_langefors,
    pattern_design,
    powder_factor,
    spacing_from_burden,
    stemming_length,
    subgrade_drilling,
)


class TestBurdenLangefors:
    """Tests for Langefors burden formula."""

    def test_positive(self):
        """Burden should be positive."""
        b = burden_langefors(89, 1200)
        assert b > 0

    def test_larger_diameter_larger_burden(self):
        """Larger D → larger burden."""
        b_small = burden_langefors(76, 1200)
        b_large = burden_langefors(115, 1200)
        assert b_large > b_small


class TestBurdenKonya:
    """Tests for Konya burden formula."""

    def test_known_value(self):
        """D=89mm, rho_e=1.2, rho_r=2.65 → B ≈ 2.4m."""
        b = burden_konya(89, 1.2, 2.65)
        assert b == pytest.approx(2.4, rel=0.1)

    def test_positive(self):
        """Burden should be positive."""
        b = burden_konya(89, 1.2, 2.65)
        assert b > 0


class TestSpacingFromBurden:
    """Tests for spacing calculation."""

    def test_default_ratio(self):
        """S = 1.15 * B."""
        s = spacing_from_burden(3.0)
        assert s == pytest.approx(3.45, rel=0.01)

    def test_custom_ratio(self):
        """S = ratio * B."""
        s = spacing_from_burden(3.0, ratio=1.25)
        assert s == pytest.approx(3.75, rel=0.01)


class TestStemmingLength:
    """Tests for stemming length."""

    def test_known(self):
        """T = 0.7 * B."""
        t = stemming_length(3.0)
        assert t == pytest.approx(2.1, rel=0.01)


class TestSubgradeDrilling:
    """Tests for subgrade drilling."""

    def test_known(self):
        """J = 0.3 * B."""
        j = subgrade_drilling(3.0)
        assert j == pytest.approx(0.9, rel=0.01)


class TestPowderFactor:
    """Tests for powder factor calculation."""

    def test_positive(self):
        """PF should be positive."""
        pf = powder_factor(1200, 89, 3.0, 3.5, 10.0, 2.1, 0.9)
        assert pf > 0

    def test_typical_range(self):
        """PF should be in typical range 0.2-1.0 kg/m³."""
        pf = powder_factor(1200, 89, 3.0, 3.5, 10.0, 2.1, 0.9)
        assert 0.1 < pf < 2.0


class TestPatternDesign:
    """Tests for complete pattern design."""

    def test_all_keys(self):
        """Should return all design parameters."""
        result = pattern_design(89, 1.2, 2.65, 10.0)
        assert "burden" in result
        assert "spacing" in result
        assert "stemming" in result
        assert "subdrill" in result
        assert "powder_factor" in result

    def test_positive_values(self):
        """All values should be positive."""
        result = pattern_design(89, 1.2, 2.65, 10.0)
        assert result["burden"] > 0
        assert result["spacing"] > 0
        assert result["powder_factor"] > 0
