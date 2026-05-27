"""문자열 목록 일괄 번역 API 단위 테스트."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.config.app_factory import create_app
from app.config.runtime import load_runtime_context
from app.services.ops import translate_text_list_with_gemini


def _app_client(monkeypatch: pytest.MonkeyPatch):
    ctx = replace(load_runtime_context(), client=object())
    monkeypatch.setattr("app.config.runtime.load_runtime_context", lambda: ctx)
    return TestClient(create_app(ctx))


def test_translate_text_list_v1_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(self, source_lang: str, target_lang: str, texts: list[str]):
        assert source_lang == "ko"
        assert target_lang == "en"
        assert texts == ["베이컨", "소고기 패티"]
        return ["Bacon", "Beef patty"]

    monkeypatch.setattr(
        "app.services.live_service.LiveService.translate_text_list",
        _fake,
    )
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/python/translations/list",
            json={
                "sourceLang": "ko",
                "targetLang": "en",
                "ingredients": [
                    {"ingredientCode": "AI_AB12CD34", "text": "베이컨"},
                    {"ingredientCode": "AI_CD34EF56", "text": "소고기 패티"},
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["results"] == [
        {"ingredientCode": "AI_AB12CD34", "translatedText": "Bacon"},
        {"ingredientCode": "AI_CD34EF56", "translatedText": "Beef patty"},
    ]


def test_translate_text_list_native_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(self, source_lang: str, target_lang: str, texts: list[str]):
        return ["tofu", "green onion"]

    monkeypatch.setattr(
        "app.services.live_service.LiveService.translate_text_list",
        _fake,
    )
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/translations/list",
            json={
                "sourceLang": "ko",
                "targetLang": "en",
                "ingredients": [
                    {"ingredientCode": "AI_EF56GH78", "text": "두부"},
                    {"ingredientCode": "AI_GH78IJ90", "text": "대파"},
                ],
            },
        )
    assert resp.status_code == 200
    assert resp.json() == {
        "results": [
            {"ingredientCode": "AI_EF56GH78", "translatedText": "tofu"},
            {"ingredientCode": "AI_GH78IJ90", "translatedText": "green onion"},
        ]
    }


def test_translate_text_list_gemini_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(self, source_lang: str, target_lang: str, texts: list[str]):
        raise genai_errors.ClientError(503, {"error": "unavailable"}, None)

    monkeypatch.setattr(
        "app.services.live_service.LiveService.translate_text_list",
        _fail,
    )
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/translations/list",
            json={
                "sourceLang": "ko",
                "targetLang": "en",
                "ingredients": [{"ingredientCode": "AI_A1", "text": "김치"}],
            },
        )
    assert resp.status_code == 502
    assert resp.json().get("code") == "PYM_502"


def test_translate_text_list_with_gemini_parses_response() -> None:
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(
        text='{"translatedTexts": ["kimchi", "pork", "tofu"]}'
    )
    result = translate_text_list_with_gemini(
        client, "gemini-2.5-flash", "ko", "en", ["김치", "돼지고기", "두부"]
    )
    assert result == ["kimchi", "pork", "tofu"]


def test_translate_text_list_rejects_empty_item() -> None:
    from pydantic import ValidationError
    from app.schemas.api_models import TextListTranslationRequest

    with pytest.raises(ValidationError, match="비어"):
        TextListTranslationRequest(
            sourceLang="ko",
            targetLang="en",
            ingredients=[
                {"ingredientCode": "AI_A1", "text": "김치"},
                {"ingredientCode": "AI_A2", "text": "  "},
            ],
        )
