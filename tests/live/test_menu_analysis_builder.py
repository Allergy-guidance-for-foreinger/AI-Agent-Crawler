from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.api_models import PythonMenuAnalysisResponse, PythonMenuAnalysisResultDto
from app.services.menu_analysis_builder import (
    build_menu_analysis_failed_result,
    build_menu_analysis_success_result,
    strip_internal_analysis_fields,
)


def test_build_menu_analysis_success_result_matches_pydantic_dto():
    raw = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=1,
            menu_name="김치찌개",
            model_name="gemini",
            model_version="gemini-2.5-flash",
            analysis={
                "ingredientsKo": [
                    {"name": "김치", "allergen": None},
                    {"name": "돼지고기", "allergen": "돼지고기"},
                    {"name": "간장", "allergen": "대두"},
                ],
                "allergensKo": [{"name": "대두", "reason": "간장"}, {"name": "돼지고기", "reason": ""}],
                "spicyLevel": 3,
            },
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.menuId == 1
    assert [i.ingredientName for i in row.ingredients] == ["김치", "돼지고기", "간장"]
    assert [i.ingredientCode for i in row.ingredients] == [None, "PORK", "SOYBEAN"]
    assert {a.allergyCode for a in row.allergies} == {"SOYBEAN", "PORK"}
    assert row.spicyLevel == 3
    assert "analyzedAt" not in raw
    assert "unmappedAllergenNames" not in raw


def test_build_menu_analysis_keeps_free_names_and_allergens_without_alias():
    raw = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=10,
            menu_name="제육볶음",
            model_name="gemini",
            model_version="gemini-2.5-flash",
            analysis={
                "ingredientsKo": [
                    {"name": "돼지고기 앞다리살", "allergen": "돼지고기"},
                    {"name": "양파", "allergen": None},
                    {"name": "고추장", "allergen": None},
                ],
                "allergensKo": [{"name": "돼지고기", "reason": "주재료"}],
                "spicyLevel": 4,
            },
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert [item.ingredientName for item in row.ingredients] == [
        "돼지고기 앞다리살",
        "양파",
        "고추장",
    ]
    assert row.ingredients[0].ingredientCode == "PORK"
    assert row.ingredients[1].ingredientCode is None
    assert len(row.allergies) == 1
    assert row.allergies[0].allergyCode == "PORK"


def test_allergies_remain_when_ingredient_alias_missing():
    raw = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=11,
            menu_name="특수메뉴",
            model_name="gemini",
            model_version="x",
            analysis={
                "ingredientsKo": [{"name": "알 수 없는 부위", "allergen": None}],
                "allergensKo": [{"name": "돼지고기", "reason": "추정"}],
                "spicyLevel": 1,
            },
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.ingredients[0].ingredientName == "알 수 없는 부위"
    assert row.ingredients[0].ingredientCode is None
    assert row.allergies[0].allergyCode == "PORK"


def test_build_menu_analysis_failed_result_matches_pydantic_dto():
    raw = build_menu_analysis_failed_result(
        menu_id=2,
        menu_name="테스트",
        model_name="gemini",
        model_version="gemini-2.5-flash",
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.status == "FAILED"
    assert row.ingredients == []
    assert row.allergies == []
    assert row.spicyLevel is None


def test_response_wrapper_accepts_results_list():
    success = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=3,
            menu_name="계란찜",
            model_name="gemini",
            model_version="x",
            analysis={"ingredientsKo": ["계란"], "allergensKo": [{"name": "난류", "reason": ""}], "spicyLevel": 2},
        )
    )
    parsed = PythonMenuAnalysisResponse.model_validate({"results": [success]})
    assert len(parsed.results) == 1


def test_success_without_spicy_level_is_null():
    raw = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=4,
            menu_name="흰밥",
            model_name="gemini",
            model_version="x",
            analysis={"ingredientsKo": ["쌀"], "allergensKo": []},
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.spicyLevel is None


def test_success_with_spicy_zero_is_not_null():
    raw = strip_internal_analysis_fields(
        build_menu_analysis_success_result(
            menu_id=5,
            menu_name="흰밥",
            model_name="gemini",
            model_version="x",
            analysis={"ingredientsKo": ["쌀"], "allergensKo": [], "spicyLevel": 0},
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.spicyLevel == 0


def test_ingredient_requires_name():
    with pytest.raises(ValidationError) as exc_info:
        PythonMenuAnalysisResultDto.model_validate(
            {
                "menuId": 1,
                "menuName": "x",
                "status": "SUCCESS",
                "spicyLevel": 0,
                "modelName": "gemini",
                "modelVersion": "x",
                "ingredients": [{"ingredientCode": "EGG", "confidence": 0.9}],
            }
        )
    errors = exc_info.value.errors()
    assert any(error.get("loc") == ("ingredients", 0, "ingredientName") for error in errors)
