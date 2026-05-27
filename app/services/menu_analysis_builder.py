"""Gemini 메뉴 분석 → Spring/PythonMenuAnalysisResultDto 응답 조립."""

from __future__ import annotations

from typing import Any

from app.services.allergen_mapping import build_ingredient_results
from app.services.ops import parse_spicy_level

DEFAULT_ALLERGEN_CONFIDENCE = 0.8


def build_menu_analysis_success_result(
    *,
    menu_id: int,
    menu_name: str,
    model_name: str,
    model_version: str,
    analysis: dict[str, Any],
    allergen_confidence: float = DEFAULT_ALLERGEN_CONFIDENCE,
) -> dict[str, Any]:
    """analyze_food_text JSON → 메뉴 분석 성공 결과 DTO dict."""
    ingredient_results = build_ingredient_results(analysis.get("ingredientsKo") or [])
    # allergies는 ingredients에 실제 등장한 알레르기 코드만 파생해 정합성을 보장합니다.
    dedup: set[str] = set()
    allergy_results: list[dict[str, Any]] = []
    for item in ingredient_results:
        code = str(item.get("ingredientCode") or "").strip()
        if not code or code in dedup:
            continue
        dedup.add(code)
        allergy_results.append({"allergyCode": code, "confidence": allergen_confidence})
    unmapped_allergens: list[str] = []
    spicy = parse_spicy_level(
        analysis.get("spicyLevel") if analysis.get("spicyLevel") is not None else analysis.get("spicy_level")
    )
    reason = analysis.get("reasonKo") or analysis.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    return {
        "menuId": menu_id,
        "menuName": menu_name,
        "status": "SUCCESS",
        "spicyLevel": spicy,
        "reason": reason,
        "modelName": model_name,
        "modelVersion": model_version,
        "ingredients": ingredient_results,
        "allergies": allergy_results,
        "_unmappedAllergenNames": unmapped_allergens,
    }


def build_menu_analysis_failed_result(
    *,
    menu_id: int,
    menu_name: str,
    model_name: str,
    model_version: str,
    reason: str,
) -> dict[str, Any]:
    """메뉴 분석 실패 결과 DTO dict."""
    return {
        "menuId": menu_id,
        "menuName": menu_name,
        "status": "FAILED",
        "spicyLevel": None,
        "reason": reason[:300],
        "modelName": model_name,
        "modelVersion": model_version,
        "ingredients": [],
        "allergies": [],
    }


def strip_internal_analysis_fields(result: dict[str, Any]) -> dict[str, Any]:
    """API 응답용 dict에서 내부 필드 제거."""
    return {k: v for k, v in result.items() if not str(k).startswith("_")}
