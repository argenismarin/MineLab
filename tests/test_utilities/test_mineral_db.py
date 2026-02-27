"""Tests for minelab.utilities.mineral_db."""

import pytest

from minelab.utilities.mineral_db import (
    MINERAL_DB,
    get_mineral,
    get_sg,
    search_minerals,
)


class TestMineralDB:
    def test_db_has_at_least_50_minerals(self):
        assert len(MINERAL_DB) >= 50

    def test_all_entries_have_required_keys(self):
        required = {"name", "formula", "sg", "hardness", "crystal_system"}
        for key, mineral in MINERAL_DB.items():
            missing = required - set(mineral.keys())
            assert not missing, (
                f"Mineral '{key}' missing keys: {missing}"
            )

    def test_sg_values_positive(self):
        for key, mineral in MINERAL_DB.items():
            assert mineral["sg"] > 0, f"Mineral '{key}' has non-positive SG"

    def test_hardness_in_range(self):
        for key, mineral in MINERAL_DB.items():
            assert 0 < mineral["hardness"] <= 10, (
                f"Mineral '{key}' hardness out of range"
            )


class TestGetMineral:
    def test_known_mineral_lowercase(self):
        m = get_mineral("quartz")
        assert m is not None
        assert m["formula"] == "SiO2"

    def test_known_mineral_mixed_case(self):
        m = get_mineral("Pyrite")
        assert m is not None
        assert m["sg"] == pytest.approx(5.02)

    def test_unknown_mineral(self):
        assert get_mineral("kryptonite") is None

    def test_leading_trailing_spaces(self):
        m = get_mineral("  gold  ")
        assert m is not None
        assert m["formula"] == "Au"


class TestGetSG:
    def test_quartz_sg(self):
        assert get_sg("quartz") == pytest.approx(2.65)

    def test_galena_sg(self):
        assert get_sg("galena") == pytest.approx(7.6)

    def test_gold_sg(self):
        assert get_sg("gold") == pytest.approx(19.3)

    def test_unknown_returns_none(self):
        assert get_sg("unobtanium") is None


class TestSearchMinerals:
    def test_search_by_name(self):
        results = search_minerals("pyrite")
        assert len(results) >= 1
        names = [r["name"].lower() for r in results]
        assert "pyrite" in names

    def test_search_by_formula_cu(self):
        results = search_minerals("Cu")
        assert len(results) >= 3  # chalcopyrite, bornite, chalcocite, etc.

    def test_search_case_insensitive(self):
        r1 = search_minerals("QUARTZ")
        r2 = search_minerals("quartz")
        assert len(r1) == len(r2)

    def test_search_no_results(self):
        results = search_minerals("zzz_nonexistent_zzz")
        assert results == []

    def test_search_partial_name(self):
        results = search_minerals("chalco")
        names = [r["name"].lower() for r in results]
        assert any("chalco" in n for n in names)
