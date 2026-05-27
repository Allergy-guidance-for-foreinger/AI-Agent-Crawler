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
        return ["kimchi", "pork"]

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
                "text": ["김치", "돼지고기"],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == ["kimchi", "pork"]


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
                "text": ["두부", "대파"],
            },
        )
    assert resp.status_code == 200
    assert resp.json() == ["tofu", "green onion"]


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
            json={"sourceLang": "ko", "targetLang": "en", "text": ["김치"]},
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
    with pytest.raises(ValueError, match="비어"):
        from app.schemas.api_models import TextListTranslationRequest

        TextListTranslationRequest(
            sourceLang="ko",
            targetLang="en",
            text=["김치", "  "],
        )
