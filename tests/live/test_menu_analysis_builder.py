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
                "ingredientsKo": ["김치", "돼지고기"],
                "allergensKo": [{"name": "대두", "reason": "두부"}],
                "spicyLevel": 3,
            },
        )
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.menuId == 1
    assert len(row.ingredients) == 2
    assert row.ingredients[0].ingredientName == "김치"
    assert row.ingredients[0].ingredientCode is None
    assert row.ingredients[1].ingredientCode == "PORK"
    assert row.allergies[0].allergyCode == "SOYBEAN"
    assert row.spicyLevel == 3
    assert "analyzedAt" not in raw
    assert "unmappedAllergenNames" not in raw


def test_build_menu_analysis_failed_result_matches_pydantic_dto():
    raw = build_menu_analysis_failed_result(
        menu_id=2,
        menu_name="테스트",
        model_name="gemini",
        model_version="gemini-2.5-flash",
        reason="503 UNAVAILABLE",
    )
    row = PythonMenuAnalysisResultDto.model_validate(raw)
    assert row.status == "FAILED"
    assert row.ingredients == []
    assert row.allergies == []
    assert row.spicyLevel == 1


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


def test_ingredient_requires_name():
    with pytest.raises(ValidationError) as exc_info:
        PythonMenuAnalysisResultDto.model_validate(
            {
                "menuId": 1,
                "menuName": "x",
                "status": "SUCCESS",
                "spicyLevel": 1,
                "modelName": "gemini",
                "modelVersion": "x",
                "ingredients": [{"ingredientCode": "EGG", "confidence": 0.9}],
            }
        )
    errors = exc_info.value.errors()
    assert any(error.get("loc") == ("ingredients", 0, "ingredientName") for error in errors)
