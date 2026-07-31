from __future__ import annotations

from datetime import date

from app.domain.entities import MenuCrawlQuery
from app.repositories.ai_repository import AIRepository
from app.repositories.crawl_repository import CrawlRepository
from app.repositories.spring_repository import SpringRepository


def test_ai_repository_map_ingredient_code_returns_known_code():
    repo = AIRepository()
    assert repo.map_ingredient_code("난류") == "EGG"


def test_crawl_repository_load_menu_table_for_source_delegates(monkeypatch):
    repo = CrawlRepository()

    def fake_loader(*, cafeteria_name, source_url):
        assert cafeteria_name == "일품식당"
        assert source_url == "https://www.kumoh.ac.kr/ko/restaurant01.do"
        return {"ok": True}

    monkeypatch.setattr(
        "app.repositories.crawl_repository.load_menu_table_for_source",
        fake_loader,
    )

    out = repo.load_menu_table_for_source(
        MenuCrawlQuery(
            cafeteria_name="학생식당",
            source_url="https://www.kumoh.ac.kr/ko/restaurant01.do",
        )
    )
    assert out == {"ok": True}


def test_crawl_repository_crawl_daily_meals_delegates(monkeypatch):
    repo = CrawlRepository()
    start = date(2026, 7, 27)
    end = date(2026, 8, 1)

    def fake_crawl(*, cafeteria_name, source_url, start, end):
        assert cafeteria_name == "정보센터식당"
        assert "shop_sqno=35" in source_url
        assert start == date(2026, 7, 27)
        assert end == date(2026, 8, 1)
        return [{"mealDate": start.isoformat(), "mealType": "LUNCH", "menus": []}]

    monkeypatch.setattr(
        "app.repositories.crawl_repository.crawl_daily_meals",
        fake_crawl,
    )

    out = repo.crawl_daily_meals(
        cafeteria_name="정보센터식당",
        source_url="https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
        start=start,
        end=end,
    )
    assert out[0]["mealType"] == "LUNCH"


def test_spring_repository_post_json_delegates(monkeypatch):
    repo = SpringRepository()

    class DummyResponse:
        status_code = 200

    def fake_post_json(*, url, payload, token, api_key):
        assert url == "https://spring.example.com/endpoint"
        assert payload == {"hello": "world"}
        assert token == "token"
        assert api_key == "api-key"
        return DummyResponse()

    monkeypatch.setattr(
        "app.repositories.spring_repository.post_json",
        fake_post_json,
    )

    out = repo.post_json(
        url="https://spring.example.com/endpoint",
        payload={"hello": "world"},
        token="token",
        api_key="api-key",
    )
    assert out.status_code == 200
