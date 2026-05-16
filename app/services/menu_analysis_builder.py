"""Gemini 메뉴 분석 → Spring/PythonMenuAnalysisResultDto 응답 조립."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.services.allergen_mapping import build_allergy_results, build_ingredient_results
from app.services.ops import clamp_spicy_level

DEFAULT_ALLERGEN_CONFIDENCE = 0.8


def build_menu_analysis_success_result(
    *,
    menu_id: int,
    menu_name: str,
    model_name: str,
    model_version: str,
    analyzed_at: datetime | str,
    analysis: dict[str, Any],
    allergen_confidence: float = DEFAULT_ALLERGEN_CONFIDENCE,
) -> dict[str, Any]:
    """analyze_food_text JSON → 메뉴 분석 성공 결과 DTO dict."""
    ingredient_results = build_ingredient_results(analysis.get("ingredientsKo") or [])
    allergy_results, unmapped_allergens = build_allergy_results(
        analysis.get("allergensKo") or [],
        fallback_confidence=allergen_confidence,
    )
    spicy = clamp_spicy_level(
        analysis.get("spicyLevel") if analysis.get("spicyLevel") is not None else analysis.get("spicy_level")
    )
    return {
        "menuId": menu_id,
        "menuName": menu_name,
        "status": "SUCCESS",
        "reason": None,
        "modelName": model_name,
        "modelVersion": model_version,
        "analyzedAt": analyzed_at,
        "ingredients": ingredient_results,
        "allergies": allergy_results,
        "unmappedAllergenNames": unmapped_allergens,
        "spicyLevel": spicy,
    }


def build_menu_analysis_failed_result(
    *,
    menu_id: int,
    menu_name: str,
    model_name: str,
    model_version: str,
    analyzed_at: datetime | str,
    reason: str,
) -> dict[str, Any]:
    """메뉴 분석 실패 결과 DTO dict."""
    return {
        "menuId": menu_id,
        "menuName": menu_name,
        "status": "FAILED",
        "reason": reason[:300],
        "modelName": model_name,
        "modelVersion": model_version,
        "analyzedAt": analyzed_at,
        "ingredients": [],
        "allergies": [],
        "unmappedAllergenNames": [],
        "spicyLevel": clamp_spicy_level(None),
    }
