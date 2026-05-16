from __future__ import annotations

from app.services.allergen_mapping import (
    build_allergy_results,
    build_ingredient_results,
    map_allergy_code,
    map_ingredient_code,
)


def test_map_allergy_code_exact_alias():
    assert map_allergy_code("난류") == "EGG"
    assert map_allergy_code("콩") == "SOYBEAN"
    assert map_allergy_code("대두") == "SOYBEAN"


def test_map_allergy_code_rejects_korean_substring_false_positive():
    assert map_allergy_code("치킨가라아게") is None
    assert map_ingredient_code("치킨가라아게") is None
    assert map_allergy_code("돼지고기 또는 해산물 (바지락, 새우 등)") is None


def test_map_allergy_code_accepts_exact_standard_labels():
    assert map_allergy_code("게") == "CRAB"
    assert map_allergy_code("새우") == "SHRIMP"
    assert map_allergy_code("가라아게") is None


def test_map_allergy_code_rejects_non_standard_labels():
    assert map_allergy_code("생선") is None
    assert map_allergy_code("어류") is None


def test_build_ingredient_results_keeps_all_names():
    rows = build_ingredient_results(["김치", "돼지고기", "파"])
    assert [r["ingredientName"] for r in rows] == ["김치", "돼지고기", "파"]
    assert rows[0]["ingredientCode"] is None
    assert rows[1]["ingredientCode"] == "PORK"


def test_build_allergy_results_and_unmapped():
    allergies, unmapped = build_allergy_results(
        [
            {"name": "대두", "reason": "두부"},
            {"name": "생선", "reason": "멸치"},
        ]
    )
    assert len(allergies) == 1
    assert allergies[0]["allergyCode"] == "SOYBEAN"
    assert unmapped == ["생선"]
