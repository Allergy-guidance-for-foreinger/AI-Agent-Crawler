"""금오·경북대 급식 크롤링/전송 도메인 모듈."""

from app.domain.crawler.knu_menu import (
    KNU_HOSTS,
    SHOP_NAMES,
    build_knu_daily_meals,
    is_knu_host,
    parse_knu_week_html,
)
from app.domain.crawler.kumoh_menu import URLS, fetch_html, load_menus, normalize_kumoh_cafeteria_name

__all__ = [
    "KNU_HOSTS",
    "SHOP_NAMES",
    "URLS",
    "build_knu_daily_meals",
    "fetch_html",
    "is_knu_host",
    "load_menus",
    "normalize_kumoh_cafeteria_name",
    "parse_knu_week_html",
]
