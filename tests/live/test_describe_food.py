"""음식명 영어 설명 API 단위 테스트."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dataclasses import replace

from app.config.app_factory import create_app
from app.config.runtime import load_runtime_context


def _app_client(monkeypatch: pytest.MonkeyPatch):
    ctx = replace(load_runtime_context(), client=object())
    monkeypatch.setattr("app.config.runtime.load_runtime_context", lambda: ctx)
    return TestClient(create_app(ctx))


def test_describe_menu_v1_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_describe(self, menu_id: int, menu_name: str):
        return {
            "menuId": menu_id,
            "menuName": menu_name,
            "description": "김치찌개는 얼큰한 한국 찌개입니다.",
            "modelName": "gemini",
            "modelVersion": "gemini-2.5-flash",
        }

    monkeypatch.setattr(
        "app.services.live_service.LiveService.describe_menu",
        _fake_describe,
    )
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/python/menus/describe",
            json={"menuId": 101, "menuName": "김치찌개"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["menuId"] == 101
    assert body["data"]["menuName"] == "김치찌개"
    assert "김치찌개" in body["data"]["description"]


def test_describe_menu_native_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_describe(self, menu_id: int, menu_name: str):
        return {
            "menuId": menu_id,
            "menuName": menu_name,
            "description": "돈까스는 돼지고기를 튀긴 일본식 요리입니다.",
            "modelName": "gemini",
            "modelVersion": "gemini-2.5-flash",
        }

    monkeypatch.setattr(
        "app.services.live_service.LiveService.describe_menu",
        _fake_describe,
    )
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/menus/describe",
            json={"menuId": 102, "menuName": "돈까스"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "success" not in body
    assert body["menuId"] == 102
    assert body["menuName"] == "돈까스"
    assert body["description"]
