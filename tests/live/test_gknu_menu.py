"""국립경국대학교(www.gknu.ac.kr) 식단 파서 단위 테스트."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from app.domain.crawler.gknu_menu import (
    build_gknu_daily_meals,
    food_view_url,
    parse_gknu_day_html,
    parse_gknu_western_html,
    resolve_gknu_cafeteria_name,
)
from app.services.ops import DEFAULT_SOURCE_ALLOWLIST, _validate_source_url, crawl_daily_meals

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gknu"


@pytest.fixture
def view_73_html() -> str:
    return (FIXTURES / "view_73.html").read_text(encoding="utf-8")


@pytest.fixture
def view_21_html() -> str:
    return (FIXTURES / "view_21.html").read_text(encoding="utf-8")


@pytest.fixture
def view_98_html() -> str:
    return (FIXTURES / "view_98.html").read_text(encoding="utf-8")


@pytest.fixture
def western_html() -> str:
    return (FIXTURES / "western_317.html").read_text(encoding="utf-8")


def _public_addrinfo(host, port, *args, **kwargs):
    return [(None, None, None, None, ("8.8.8.8", port or 443))]


class TestGknuDayParse:
    def test_iroom_lunch_and_breakfast(self, view_73_html: str):
        meals = parse_gknu_day_html(view_73_html, meal_date=date(2026, 7, 24))
        assert {m["mealType"] for m in meals} == {"BREAKFAST", "LUNCH"}
        breakfast = next(m for m in meals if m["mealType"] == "BREAKFAST")
        assert breakfast["menus"][0]["cornerName"] == "조식"
        assert "[천원의 아침밥]" in breakfast["menus"][0]["menuName"]

        lunch = next(m for m in meals if m["mealType"] == "LUNCH")
        names = [item["menuName"] for item in lunch["menus"]]
        assert names[0] == "[천원의 브런치]"
        assert "흑미밥" in names
        assert "꿔바로우" in names
        assert all(item["cornerName"] == "중식" for item in lunch["menus"])

    def test_closed_meals_filtered(self, view_98_html: str):
        meals = parse_gknu_day_html(view_98_html, meal_date=date(2026, 7, 24))
        assert {m["mealType"] for m in meals} == {"LUNCH"}
        lunch = meals[0]
        assert "참치생채소비빔밥" in lunch["menus"][0]["menuName"]
        assert not any("미" in item["menuName"] and "영" in item["menuName"] for item in lunch["menus"])

    def test_empty_dinner_skipped(self, view_21_html: str):
        meals = parse_gknu_day_html(view_21_html, meal_date=date(2026, 7, 24))
        assert "DINNER" not in {m["mealType"] for m in meals}

    def test_empty_html_returns_empty(self):
        assert parse_gknu_day_html("", meal_date=date(2026, 7, 24)) == []


class TestGknuWesternParse:
    def test_western_menu_list(self, western_html: str):
        menus = parse_gknu_western_html(western_html)
        assert len(menus) >= 10
        assert menus[0]["cornerName"] == "양식코너"
        assert menus[0]["menuName"] == "제주흑돼지김치찌개"
        assert all(item["displayOrder"] == i + 1 for i, item in enumerate(menus))
        assert not any("," in item["menuName"] for item in menus)


class TestGknuHelpers:
    def test_food_view_url(self):
        assert food_view_url(73, date(2026, 7, 24)).endswith("manage_idx=73&memo5=2026-07-24")

    def test_resolve_rejects_name_mismatch(self):
        with pytest.raises(RuntimeError, match="일치하지 않습니다"):
            resolve_gknu_cafeteria_name(
                "채움관(안동, 교직원식당)",
                "https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
            )

    def test_default_allowlist_includes_gknu(self):
        assert "www.gknu.ac.kr" in DEFAULT_SOURCE_ALLOWLIST
        assert "gknu.ac.kr" in DEFAULT_SOURCE_ALLOWLIST


class TestGknuBuildDailyMeals:
    def test_build_fetches_per_day(self, view_73_html: str):
        with patch(
            "app.domain.crawler.gknu_menu.fetch_html",
            return_value=view_73_html,
        ) as mocked:
            meals = build_gknu_daily_meals(
                cafeteria_name="이룸관(안동, 학생식당)",
                source_url="https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
                start=date(2026, 7, 23),
                end=date(2026, 7, 24),
            )
        assert mocked.call_count == 2
        urls = [call.args[0] for call in mocked.call_args_list]
        assert any("memo5=2026-07-23" in u for u in urls)
        assert any("memo5=2026-07-24" in u for u in urls)
        assert any(m["mealDate"] == "2026-07-24" and m["mealType"] == "LUNCH" for m in meals)

    def test_partial_day_fetch_failure_keeps_success(self, view_73_html: str):
        def fake_fetch(url: str, **kwargs):
            if "2026-07-24" in url:
                raise requests.exceptions.Timeout("boom")
            return view_73_html

        with patch("app.domain.crawler.gknu_menu.fetch_html", side_effect=fake_fetch):
            meals = build_gknu_daily_meals(
                cafeteria_name="이룸관(안동, 학생식당)",
                source_url="https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
                start=date(2026, 7, 23),
                end=date(2026, 7, 24),
            )
        assert any(m["mealDate"] == "2026-07-23" for m in meals)
        assert not any(m["mealDate"] == "2026-07-24" for m in meals)

    def test_western_replicates_across_dates(self, western_html: str):
        with patch("app.domain.crawler.gknu_menu.fetch_html", return_value=western_html) as mocked:
            meals = build_gknu_daily_meals(
                cafeteria_name="양식코너(안동)",
                source_url="https://www.gknu.ac.kr/main/html.do?menu_idx=317",
                start=date(2026, 7, 23),
                end=date(2026, 7, 24),
            )
        assert mocked.call_count == 1
        assert len(meals) == 2
        assert {m["mealType"] for m in meals} == {"LUNCH"}
        assert meals[0]["menus"][0]["menuName"] == meals[1]["menus"][0]["menuName"]

    def test_crawl_daily_meals_routes_gknu(self, view_73_html: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CRAWL_SOURCE_ALLOWLIST", raising=False)
        monkeypatch.setattr("app.services.ops.socket.getaddrinfo", _public_addrinfo)
        with patch(
            "app.domain.crawler.gknu_menu.fetch_html",
            return_value=view_73_html,
        ):
            meals = crawl_daily_meals(
                cafeteria_name="이룸관(안동, 학생식당)",
                source_url="https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
                start=date(2026, 7, 24),
                end=date(2026, 7, 24),
            )
        assert {m["mealType"] for m in meals} == {"BREAKFAST", "LUNCH"}

    def test_blank_allowlist_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CRAWL_SOURCE_ALLOWLIST", " , , ")
        monkeypatch.setattr("app.services.ops.socket.getaddrinfo", _public_addrinfo)
        _validate_source_url(
            "https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82"
        )
