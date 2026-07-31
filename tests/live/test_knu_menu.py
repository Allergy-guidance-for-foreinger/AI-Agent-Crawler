"""경북대(coop.knu.ac.kr) 식단 파서 단위 테스트."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from app.domain.crawler.knu_menu import (
    build_knu_daily_meals,
    parse_knu_week_html,
    resolve_knu_cafeteria_name,
    week_mondays_covering,
    with_sel_date,
)
from app.services.ops import DEFAULT_SOURCE_ALLOWLIST, _validate_source_url, crawl_daily_meals

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "knu"


@pytest.fixture
def shop35_html() -> str:
    return (FIXTURES / "shop_35.html").read_text(encoding="utf-8")


@pytest.fixture
def shop36_html() -> str:
    return (FIXTURES / "shop_36.html").read_text(encoding="utf-8")


def _public_addrinfo(host, port, *args, **kwargs):
    # 사설/예약 대역이 아닌 공인 IP로 DNS 결과를 흉내 냅니다.
    return [(None, None, None, None, ("8.8.8.8", port or 443))]


class TestKnuWeekParse:
    def test_info_center_lunch_and_dinner(self, shop35_html: str):
        meals = parse_knu_week_html(
            shop35_html,
            cafeteria_name="정보센터식당",
            start=date(2026, 7, 27),
            end=date(2026, 8, 1),
        )
        assert len(meals) == 10  # 월~금 x 중식/석식
        monday_lunch = next(
            m for m in meals if m["mealDate"] == "2026-07-27" and m["mealType"] == "LUNCH"
        )
        assert monday_lunch["menus"][0]["cornerName"] == "특식"
        assert "육전비빔국수" in monday_lunch["menus"][0]["menuName"]
        assert all("￦" not in item["menuName"] for item in monday_lunch["menus"])
        assert len(monday_lunch["menus"]) >= 8

        monday_dinner = next(
            m for m in meals if m["mealDate"] == "2026-07-27" and m["mealType"] == "DINNER"
        )
        assert "촌돼지찌개" in monday_dinner["menus"][0]["menuName"]

    def test_faculty_lunch_only(self, shop36_html: str):
        meals = parse_knu_week_html(
            shop36_html,
            cafeteria_name="복지관 교직원식당",
            start=date(2026, 7, 27),
            end=date(2026, 7, 31),
        )
        assert meals
        assert {m["mealType"] for m in meals} == {"LUNCH"}
        assert meals[0]["menus"][0]["cornerName"] == "정식"
        assert "청국장찌개" in meals[0]["menus"][0]["menuName"]

    def test_corner_only_title_uses_body(self):
        html = """
        <table class="tstyle_me"><tr>
          <th>구분</th><th>월<p class="week_t">(07/27)</p></th>
        </tr></table>
        <div class="week_table mt5">중식
          <table><tr><td>
            <div class="button_m">정식</div>
            <ul class="menu_im"><li class="first">정식<p>흑미밥<br> 된장국<br> 불고기</p><p>￦ 6,000</p></li></ul>
          </td></tr></table>
        </div>
        """
        meals = parse_knu_week_html(
            html,
            cafeteria_name="공학관교직원식당(외부업체)",
            start=date(2026, 7, 27),
            end=date(2026, 7, 27),
        )
        assert meals[0]["menus"][0]["menuName"] == "흑미밥 된장국 불고기"
        assert meals[0]["mealDate"] == "2026-07-27"

    def test_empty_html_returns_empty(self):
        assert parse_knu_week_html("", cafeteria_name="정보센터식당", start=date(2026, 7, 27), end=date(2026, 7, 27)) == []

    def test_date_range_filter(self, shop35_html: str):
        meals = parse_knu_week_html(
            shop35_html,
            cafeteria_name="정보센터식당",
            start=date(2026, 7, 31),
            end=date(2026, 7, 31),
        )
        assert {m["mealDate"] for m in meals} == {"2026-07-31"}
        assert {m["mealType"] for m in meals} == {"LUNCH", "DINNER"}


class TestKnuHelpers:
    def test_week_mondays_covering(self):
        assert week_mondays_covering(date(2026, 7, 31), date(2026, 8, 5)) == [
            date(2026, 7, 27),
            date(2026, 8, 3),
        ]

    def test_with_sel_date(self):
        url = with_sel_date(
            "https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
            date(2026, 7, 27),
        )
        assert "shop_sqno=35" in url
        assert "selDate=2026-07-27" in url

    def test_resolve_rejects_name_sqno_mismatch(self):
        with pytest.raises(RuntimeError, match="일치하지 않습니다"):
            resolve_knu_cafeteria_name(
                "복지관 교직원식당",
                "https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
            )

    def test_default_allowlist_includes_knu(self):
        assert "coop.knu.ac.kr" in DEFAULT_SOURCE_ALLOWLIST


class TestKnuBuildDailyMeals:
    def test_build_fetches_with_sel_date(self, shop35_html: str):
        with patch(
            "app.domain.crawler.knu_menu.fetch_html",
            return_value=shop35_html,
        ) as mocked:
            meals = build_knu_daily_meals(
                cafeteria_name="정보센터식당",
                source_url="https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
                start=date(2026, 7, 27),
                end=date(2026, 7, 28),
            )
        assert mocked.call_count == 1
        assert "selDate=2026-07-27" in mocked.call_args.args[0]
        assert len(meals) == 4

    def test_partial_week_fetch_failure_keeps_success(self, shop35_html: str):
        calls = {"n": 0}

        def fake_fetch(url: str):
            calls["n"] += 1
            if "2026-08-03" in url:
                raise requests.exceptions.Timeout("boom")
            return shop35_html

        with patch("app.domain.crawler.knu_menu.fetch_html", side_effect=fake_fetch):
            meals = build_knu_daily_meals(
                cafeteria_name="정보센터식당",
                source_url="https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
                start=date(2026, 7, 31),
                end=date(2026, 8, 5),
            )
        assert calls["n"] == 2
        assert any(m["mealDate"] == "2026-07-31" for m in meals)

    def test_crawl_daily_meals_routes_knu(self, shop35_html: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CRAWL_SOURCE_ALLOWLIST", raising=False)
        monkeypatch.setattr("app.services.ops.socket.getaddrinfo", _public_addrinfo)
        with patch(
            "app.domain.crawler.knu_menu.fetch_html",
            return_value=shop35_html,
        ):
            meals = crawl_daily_meals(
                cafeteria_name="정보센터식당",
                source_url="https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
                start=date(2026, 7, 27),
                end=date(2026, 7, 27),
            )
        assert len(meals) == 2
        assert {m["mealType"] for m in meals} == {"LUNCH", "DINNER"}

    def test_blank_allowlist_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CRAWL_SOURCE_ALLOWLIST", " , , ")
        monkeypatch.setattr("app.services.ops.socket.getaddrinfo", _public_addrinfo)
        _validate_source_url("https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35")
