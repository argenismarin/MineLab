"""Tests for minelab.environmental.acid_drainage."""

import pytest

from minelab.environmental.acid_drainage import (
    acid_neutralizing_capacity,
    maximum_potential_acidity,
    nag_test_classify,
    napp,
    paste_ph_prediction,
)


class TestMaximumPotentialAcidity:
    """Tests for MPA calculation."""

    def test_known_value(self):
        """2%S → MPA = 61.2 kg H2SO4/t."""
        mpa = maximum_potential_acidity(2.0)
        assert mpa == pytest.approx(61.2, rel=0.01)

    def test_zero_sulfur(self):
        """Zero sulfur → zero MPA."""
        mpa = maximum_potential_acidity(0.0)
        assert mpa == pytest.approx(0.0)


class TestAcidNeutralizingCapacity:
    """Tests for ANC calculation."""

    def test_known_value(self):
        """5% CaCO3 → ANC = 50."""
        anc = acid_neutralizing_capacity({"calcium_carbonate_pct": 5.0})
        assert anc == pytest.approx(50.0, rel=0.05)

    def test_positive(self):
        """ANC should be positive for positive carbonate."""
        anc = acid_neutralizing_capacity({"calcium_carbonate_pct": 3.0})
        assert anc > 0


class TestNAPP:
    """Tests for NAPP calculation."""

    def test_paf(self):
        """MPA=60, ANC=30 → NAPP=30 (PAF)."""
        result = napp(60, 30)
        assert result["napp"] == pytest.approx(30, rel=0.01)
        assert result["classification"] == "PAF"

    def test_naf(self):
        """MPA=20, ANC=50 → NAPP=-30 (NAF)."""
        result = napp(20, 50)
        assert result["napp"] == pytest.approx(-30, rel=0.01)
        assert result["classification"] == "NAF"


class TestNAGTestClassify:
    """Tests for NAG test classification."""

    def test_paf(self):
        """NAG pH<4.5, NAG>5 → PAF."""
        result = nag_test_classify(3.5, 10.0)
        assert result["classification"] == "PAF"

    def test_naf(self):
        """NAG pH>4.5 → NAF."""
        result = nag_test_classify(5.5, 2.0)
        assert result["classification"] == "NAF"


class TestPastePHPrediction:
    """Tests for paste pH prediction."""

    def test_low_sulfide(self):
        """Low sulfide → near-neutral pH."""
        result = paste_ph_prediction(0.1)
        assert result["predicted_ph"] > 5

    def test_high_sulfide(self):
        """High sulfide → low pH."""
        result = paste_ph_prediction(3.0)
        assert result["predicted_ph"] < 5


class TestANCCalciumMagnesium:
    """Tests for ANC using calcium and magnesium percentages."""

    def test_ca_mg_known_value(self):
        """Ca=2.0%, Mg=1.0% → ANC = (2*2.497 + 1*4.116)*10 = 91.1."""
        anc = acid_neutralizing_capacity({"calcium_pct": 2.0, "magnesium_pct": 1.0})
        assert anc == pytest.approx(91.1, rel=1e-3)

    def test_ca_mg_zero(self):
        """Zero calcium and magnesium → zero ANC."""
        anc = acid_neutralizing_capacity({"calcium_pct": 0.0, "magnesium_pct": 0.0})
        assert anc == pytest.approx(0.0)

    def test_missing_keys_raises(self):
        """Missing required keys should raise ValueError."""
        with pytest.raises(ValueError, match="calcium_carbonate_pct"):
            acid_neutralizing_capacity({"sulfur_pct": 2.0})

    def test_only_calcium_pct_missing_magnesium_raises(self):
        """Having only calcium_pct without magnesium_pct should raise ValueError."""
        with pytest.raises(ValueError, match="calcium_pct"):
            acid_neutralizing_capacity({"calcium_pct": 2.0})


class TestNAPPUncertain:
    """Tests for NAPP uncertain classification."""

    def test_equal_mpa_anc(self):
        """MPA == ANC → NAPP = 0 → Uncertain."""
        result = napp(50.0, 50.0)
        assert result["napp"] == pytest.approx(0.0)
        assert result["classification"] == "Uncertain"


class TestNAGTestClassifyUncertain:
    """Tests for NAG test uncertain classification."""

    def test_low_ph_low_nag_value(self):
        """NAG pH < 4.5 but NAG value <= 5 → Uncertain."""
        result = nag_test_classify(4.0, 3.0)
        assert result["classification"] == "Uncertain"


class TestPastePHAlkaline:
    """Tests for paste pH alkaline classification."""

    def test_high_neutralizer_alkaline(self):
        """High neutralizer → pH > 8.5 → Alkaline."""
        # raw_ph = 7 - 3*0.1 + 2*2.0 = 7 - 0.3 + 4.0 = 10.7
        result = paste_ph_prediction(0.1, neutralizer_pct=2.0)
        assert result["predicted_ph"] == pytest.approx(10.7)
        assert result["classification"] == "Alkaline"
