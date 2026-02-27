"""Tests for minelab.geomechanics.hoek_brown."""

import numpy as np
import pytest

from minelab.geomechanics.hoek_brown import (
    deformation_modulus,
    hoek_brown_intact,
    hoek_brown_parameters,
    hoek_brown_rock_mass,
    mohr_coulomb_fit,
)


class TestHoekBrownIntact:
    """Tests for intact rock Hoek-Brown criterion."""

    def test_uniaxial(self):
        """At σ3=0, σ1 = σci."""
        assert hoek_brown_intact(0, 100, 25) == 100.0

    def test_known_value(self):
        """σ3=10, σci=100, mi=25 → σ1 ≈ 183.1."""
        result = hoek_brown_intact(10, 100, 25)
        expected = 10 + 100 * np.sqrt(25 * 10 / 100 + 1)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_array_input(self):
        """Vectorized σ3."""
        sig3 = np.array([0, 5, 10, 20])
        sig1 = hoek_brown_intact(sig3, 100, 25)
        assert sig1[0] == 100.0
        assert np.all(np.diff(sig1) > 0)

    def test_invalid_sigci(self):
        """Negative σci should raise."""
        with pytest.raises(ValueError):
            hoek_brown_intact(0, -10, 25)


class TestHoekBrownParameters:
    """Tests for HB parameter calculation."""

    def test_known_values(self):
        """GSI=50, mi=25, D=0 → mb≈4.205, s≈0.0039, a≈0.506."""
        p = hoek_brown_parameters(50, 25, 0)
        assert p["mb"] == pytest.approx(4.205, rel=0.01)
        assert p["s"] == pytest.approx(0.0039, rel=0.05)
        assert p["a"] == pytest.approx(0.506, rel=0.01)

    def test_intact_rock(self):
        """GSI=100, D=0 → mb=mi, s=1, a=0.5."""
        p = hoek_brown_parameters(100, 25, 0)
        assert p["mb"] == pytest.approx(25.0, rel=0.01)
        assert p["s"] == pytest.approx(1.0, rel=0.01)
        assert p["a"] == pytest.approx(0.5, rel=0.01)

    def test_disturbed(self):
        """D=1 reduces mb and s."""
        p_und = hoek_brown_parameters(50, 25, 0)
        p_dist = hoek_brown_parameters(50, 25, 1)
        assert p_dist["mb"] < p_und["mb"]
        assert p_dist["s"] < p_und["s"]

    def test_invalid_gsi(self):
        """GSI > 100 should raise."""
        with pytest.raises(ValueError):
            hoek_brown_parameters(110, 25, 0)

    def test_invalid_d(self):
        """D > 1 should raise."""
        with pytest.raises(ValueError):
            hoek_brown_parameters(50, 25, 1.5)


class TestHoekBrownRockMass:
    """Tests for generalized rock mass HB criterion."""

    def test_uniaxial_rock_mass(self):
        """At σ3=0, σ1 = σci * s^a (rock mass UCS)."""
        p = hoek_brown_parameters(50, 25, 0)
        result = hoek_brown_rock_mass(0, 100, 50, 25, 0)
        expected = 100 * p["s"] ** p["a"]
        assert result == pytest.approx(expected, rel=1e-3)

    def test_weaker_than_intact(self):
        """Rock mass should be weaker than intact at same σ3."""
        sig3 = 10
        intact = hoek_brown_intact(sig3, 100, 25)
        mass = hoek_brown_rock_mass(sig3, 100, 50, 25, 0)
        assert mass < intact

    def test_gsi100_equals_intact(self):
        """At GSI=100, D=0, rock mass ≈ intact."""
        sig3 = 10.0
        intact = hoek_brown_intact(sig3, 100, 25)
        mass = hoek_brown_rock_mass(sig3, 100, 100, 25, 0)
        assert mass == pytest.approx(intact, rel=0.02)


class TestMohrCoulombFit:
    """Tests for Mohr-Coulomb equivalent fit."""

    def test_positive_cohesion(self):
        """Cohesion must be positive."""
        result = mohr_coulomb_fit(100, 50, 25, 0)
        assert result["cohesion"] > 0

    def test_positive_friction_angle(self):
        """Friction angle must be positive."""
        result = mohr_coulomb_fit(100, 50, 25, 0)
        assert result["friction_angle"] > 0

    def test_friction_angle_range(self):
        """Friction angle typically 20-60° for rock."""
        result = mohr_coulomb_fit(100, 50, 25, 0)
        assert 10 < result["friction_angle"] < 70

    def test_disturbed_lower_strength(self):
        """Disturbed rock should have lower cohesion."""
        r_und = mohr_coulomb_fit(100, 50, 25, 0)
        r_dist = mohr_coulomb_fit(100, 50, 25, 1)
        assert r_dist["cohesion"] < r_und["cohesion"]

    def test_custom_sig3_max(self):
        """Custom σ3 max should work."""
        result = mohr_coulomb_fit(100, 50, 25, 0, sig3_max=50)
        assert result["cohesion"] > 0


class TestDeformationModulus:
    """Tests for rock mass deformation modulus."""

    def test_with_ei(self):
        """Known: σci=100, GSI=50, D=0, Ei=50000 → Erm ≈ 15359 MPa."""
        erm = deformation_modulus(100, 50, 0, 50000)
        # Erm = Ei * (0.02 + (1-D/2)/(1+exp((60+15D-GSI)/11)))
        expected = 50000 * (0.02 + 1 / (1 + np.exp(10 / 11)))
        assert erm == pytest.approx(expected, rel=0.01)

    def test_without_ei(self):
        """Simplified equation (Ei unknown)."""
        erm = deformation_modulus(100, 50, 0)
        assert erm > 0
        assert erm < 100000

    def test_higher_gsi_higher_modulus(self):
        """Higher GSI → higher deformation modulus."""
        e50 = deformation_modulus(100, 50, 0, 50000)
        e80 = deformation_modulus(100, 80, 0, 50000)
        assert e80 > e50

    def test_disturbance_reduces(self):
        """Disturbance D=1 → lower modulus."""
        e_und = deformation_modulus(100, 50, 0, 50000)
        e_dist = deformation_modulus(100, 50, 1, 50000)
        assert e_dist < e_und

    def test_invalid_gsi(self):
        """GSI out of range should raise."""
        with pytest.raises(ValueError):
            deformation_modulus(100, 110, 0)
