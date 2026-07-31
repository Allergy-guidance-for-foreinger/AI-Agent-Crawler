from __future__ import annotations

from app.services.allergen_mapping import (
    build_allergy_results,
    build_ingredient_results,
    map_allergy_code,
    map_ingredient_code,
    merge_allergy_results,
    normalize_ingredient_name,
    parse_ingredient_item,
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


def test_normalize_keeps_name_and_maps_alias_code():
    name, code = normalize_ingredient_name("앞다리살")
    assert name == "앞다리살"
    assert code == "PORK"
    name, code = normalize_ingredient_name("김치")
    assert name == "김치"
    assert code is None


def test_parse_ingredient_item_uses_allergen_field():
    name, code = parse_ingredient_item({"name": "돼지고기 앞다리살", "allergen": "돼지고기"})
    assert name == "돼지고기 앞다리살"
    assert code == "PORK"
    name, code = parse_ingredient_item({"name": "김치", "allergen": None})
    assert name == "김치"
    assert code is None


def test_build_ingredient_results_keeps_all_free_names():
    rows = build_ingredient_results(
        [
            {"name": "김치", "allergen": None},
            {"name": "앞다리살", "allergen": "돼지고기"},
            "간장",
            "양파",
        ]
    )
    assert [r["ingredientName"] for r in rows] == ["김치", "앞다리살", "간장", "양파"]
    assert rows[0]["ingredientCode"] is None
    assert rows[1]["ingredientCode"] == "PORK"
    assert rows[2]["ingredientCode"] == "SOYBEAN"
    assert rows[3]["ingredientCode"] is None


def test_merge_allergy_results_uses_allergens_and_ingredient_codes():
    ingredients = build_ingredient_results(
        [{"name": "특수부위", "allergen": "돼지고기"}, {"name": "김치", "allergen": None}]
    )
    allergies, unmapped = merge_allergy_results(
        allergens_ko=[{"name": "대두", "reason": "간장"}, {"name": "생선", "reason": "x"}],
        ingredient_results=ingredients,
    )
    codes = {a["allergyCode"] for a in allergies}
    assert codes == {"SOYBEAN", "PORK"}
    assert unmapped == ["생선"]


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
