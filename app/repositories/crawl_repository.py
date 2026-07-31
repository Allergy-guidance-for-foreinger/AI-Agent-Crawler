"""크롤링/메뉴 데이터 접근 Repository."""

from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import urlparse

from google import genai

from app.domain.crawler.knu_menu import is_knu_host
from app.domain.crawler.kumoh_menu import normalize_kumoh_cafeteria_name
from app.domain.entities import MenuCrawlQuery
from app.common.service_ops import (
    build_daily_meals,
    crawl_daily_meals,
    load_menu_table_for_source,
    run_weekly_crawl_once,
)


class CrawlRepository:
    """크롤링/메뉴 데이터 접근 Repository."""

    def crawl_daily_meals(
        self,
        *,
        cafeteria_name: str,
        source_url: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        return crawl_daily_meals(
            cafeteria_name=cafeteria_name,
            source_url=source_url,
            start=start,
            end=end,
        )

    def load_menu_table_for_source(self, query: MenuCrawlQuery):
        cafeteria_name = query.cafeteria_name
        if not is_knu_host(urlparse(query.source_url).hostname):
            cafeteria_name = normalize_kumoh_cafeteria_name(cafeteria_name)
        return load_menu_table_for_source(
            cafeteria_name=cafeteria_name,
            source_url=query.source_url,
        )

    def build_daily_meals(
        self,
        *,
        cafeteria_name: str,
        table: Any,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        return build_daily_meals(cafeteria_name=cafeteria_name, table=table, start=start, end=end)

    def run_weekly_crawl_once(self, cfg, client: genai.Client | None) -> dict[str, Any]:
        return run_weekly_crawl_once(cfg, client)
