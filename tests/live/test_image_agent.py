"""이미지 분석 Gemini 응답 파싱 단위 테스트."""

from __future__ import annotations

from app.domain.image.agent import (
    extract_confidence_from_image_analysis,
    extract_food_name_from_image_analysis,
    extract_food_name_reason_from_image_analysis,
    extract_ingredient_names_from_image_analysis,
)


def test_extract_from_standard_schema():
    payload = {
        "음식명": "제육볶음",
        "추정_식재료": [
            {"재료": "돼지고기", "근거": "고기 조각"},
            {"재료": "고추장", "근거": "양념"},
        ],
    }
    assert extract_ingredient_names_from_image_analysis(payload) == ["돼지고기", "고추장"]


def test_extract_fallback_ingredients_ko():
    payload = {"ingredientsKo": ["돼지고기", "양파", "고추장"]}
    assert extract_ingredient_names_from_image_analysis(payload) == ["돼지고기", "양파", "고추장"]


def test_extract_fallback_string_list():
    payload = {"ingredients": ["밥", "계란"]}
    assert extract_ingredient_names_from_image_analysis(payload) == ["밥", "계란"]


def test_extract_empty():
    assert extract_ingredient_names_from_image_analysis({}) == []


def test_extract_food_name():
    assert extract_food_name_from_image_analysis({"음식명": "제육볶음"}) == "제육볶음"
    assert extract_food_name_from_image_analysis({"foodNameKo": "김치찌개"}) == "김치찌개"
    assert extract_food_name_from_image_analysis({}) is None


def test_extract_food_name_reason():
    payload = {
        "음식명": "제육볶음",
        "음식명_근거": "붉은 고추장 양념의 돼지고기와 양파가 보입니다.",
    }
    assert extract_food_name_reason_from_image_analysis(payload) == payload["음식명_근거"]
    assert extract_food_name_reason_from_image_analysis({}) is None


def test_extract_confidence():
    assert extract_confidence_from_image_analysis({"confidence": 0.81}) == 0.81
    assert extract_confidence_from_image_analysis({"confidence": 81}) == 0.81
    assert extract_confidence_from_image_analysis({}) is None
