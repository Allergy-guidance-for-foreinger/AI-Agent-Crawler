"""음식명 번역/발음 분리 단위 테스트."""

from __future__ import annotations

from app.services.ops import _parse_json_field_fallback, localize_food_name_with_gemini


def test_parse_json_field_fallback_recovers_truncated_pronunciation():
    raw = '{\n  "translation": "Spicy Pork Stir-fry",\n  "pronunciation": "Jeyuk-bok'
    assert _parse_json_field_fallback(raw, "translation") == "Spicy Pork Stir-fry"
    assert _parse_json_field_fallback(raw, "pronunciation") == "Jeyuk-bok"


def test_localize_food_name_ko_without_client():
    result = localize_food_name_with_gemini(None, "gemini-2.5-flash", "ko", "제육볶음")
    assert result == {"translation": "제육볶음", "pronunciation": "제육볶음"}
