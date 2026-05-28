"""메뉴 목록 설명 API 단위 테스트."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config.app_factory import create_app
from app.config.runtime import load_runtime_context


def _app_client(monkeypatch: pytest.MonkeyPatch):
    ctx = replace(load_runtime_context(), client=object())
    monkeypatch.setattr("app.config.runtime.load_runtime_context", lambda: ctx)
    return TestClient(create_app(ctx))


def test_describe_menus_list_v1_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(self, menus, *, lang_code: str = "ko", max_concurrency: int = 4):
        assert lang_code == "ko"
        return [
            {"menuId": 1, "description": "김치와 돼지고기를 넣고 끓인 매콤한 한국식 찌개입니다."},
            {"menuId": 2, "description": "바삭하게 튀긴 돼지고기 커틀릿에 소스를 곁들인 음식입니다."},
        ]

    monkeypatch.setattr("app.services.live_service.LiveService.describe_menus", _fake)
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/python/menus/describe/list",
            json={
                "langCode": "ko",
                "menus": [
                    {"menuId": 1, "menuName": "김치찌개"},
                    {"menuId": 2, "menuName": "돈까스"},
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["results"]) == 2
    assert body["data"]["results"][0]["menuId"] == 1


def test_describe_menus_list_native_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(self, menus, *, lang_code: str = "ko", max_concurrency: int = 4):
        return [
            {"menuId": 1, "description": "kimchi stew description"},
            {"menuId": 2, "description": "pork cutlet description"},
        ]

    monkeypatch.setattr("app.services.live_service.LiveService.describe_menus", _fake)
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/menus/describe/list",
            json={
                "langCode": "en",
                "menus": [
                    {"menuId": 1, "menuName": "김치찌개"},
                    {"menuId": 2, "menuName": "돈까스"},
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "success" not in body
    assert body["results"][1]["menuId"] == 2
    assert "description" in body["results"][1]


def test_describe_menus_list_allows_empty_description(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(self, menus, *, lang_code: str = "ko", max_concurrency: int = 4):
        return [{"menuId": 1, "description": ""}]

    monkeypatch.setattr("app.services.live_service.LiveService.describe_menus", _fake)
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/python/menus/describe/list",
            json={"langCode": "ko", "menus": [{"menuId": 1, "menuName": "김치찌개"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["results"][0]["description"] == ""


def test_describe_menus_list_runtime_error_masks_message(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(self, menus, *, lang_code: str = "ko", max_concurrency: int = 4):
        raise RuntimeError("upstream raw detail")

    monkeypatch.setattr("app.services.live_service.LiveService.describe_menus", _fail)
    with _app_client(monkeypatch) as client:
        resp = client.post(
            "/api/v1/menus/describe/list",
            json={"langCode": "ko", "menus": [{"menuId": 1, "menuName": "김치찌개"}]},
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["msg"] == "요청 처리 중 내부 오류가 발생했습니다."

