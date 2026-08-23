"""국립경국대학교(www.gknu.ac.kr) 식단 HTML/AJAX 파서."""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse

import requests
from lxml import html as lhtml
from lxml.etree import ParserError, _Element

from app.domain.crawler.knu_menu import MAX_WEEK_FETCHES, week_mondays_covering

logger = logging.getLogger(__name__)

GKNU_HOSTS = frozenset({"www.gknu.ac.kr", "gknu.ac.kr"})
GKNU_ORIGIN = "https://www.gknu.ac.kr"
FOOD_VIEW_PATH = "/main/module/foodMenu/view.do"
# 일별 AJAX는 날짜 수만큼 호출되므로 knu(주 단위)보다 짧게 잡습니다.
DAY_FETCH_TIMEOUT = 5.0
DAY_FETCH_BUDGET_SECONDS = 45.0

# menu_idx → 식당명
MENU_IDX_NAMES: dict[int, str] = {
    82: "이룸관(안동, 학생식당)",
    222: "채움관(안동, 교직원식당)",
    317: "양식코너(안동)",
    629: "학생식당(예천)",
}

# 일별 AJAX용 menu_idx → manage_idx
MENU_IDX_TO_MANAGE: dict[int, int] = {
    82: 73,
    222: 21,
    629: 98,
}

WESTERN_MENU_IDX = 317
MEAL_LABEL_TO_TYPE = {
    "조식": "BREAKFAST",
    "중식": "LUNCH",
    "석식": "DINNER",
}
_CLOSED_RE = re.compile(r"미\s*운\s*영")
_HEADER_SKIP = frozenset({"대표메뉴", "가격", "메뉴", "양식코너"})


def is_gknu_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    return hostname.lower() in GKNU_HOSTS


def parse_menu_idx(source_url: str) -> int | None:
    parsed = urlparse(source_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    raw = qs.get("menu_idx", "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def resolve_gknu_cafeteria_name(cafeteria_name: str, source_url: str) -> str:
    """menu_idx 기준 식당명을 확정하고, 요청명이 있으면 일치 여부를 검증합니다."""
    menu_idx = parse_menu_idx(source_url)
    if menu_idx is None or menu_idx not in MENU_IDX_NAMES:
        raise RuntimeError("경국대 menu_idx를 확인할 수 없거나 지원하지 않는 식당입니다.")
    expected = MENU_IDX_NAMES[menu_idx]
    name = (cafeteria_name or "").strip()
    if not name:
        return expected
    if name != expected:
        raise RuntimeError(
            f"경국대 식당명이 menu_idx와 일치하지 않습니다: "
            f"cafeteriaName={name}, expected={expected}(menu_idx={menu_idx})"
        )
    return expected


def fetch_html(url: str, *, timeout: float = DAY_FETCH_TIMEOUT, referer: str | None = None) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Agent-Crawler/1.0)"}
    if referer:
        headers["Referer"] = referer
        headers["X-Requested-With"] = "XMLHttpRequest"
    res = requests.get(
        url,
        timeout=timeout,
        allow_redirects=False,
        headers=headers,
    )
    if 300 <= res.status_code < 400:
        raise requests.exceptions.RequestException("redirect is not allowed for source_url")
    res.raise_for_status()
    res.encoding = res.apparent_encoding or "utf-8"
    return res.text


def food_view_url(manage_idx: int, meal_date: date) -> str:
    return (
        f"{GKNU_ORIGIN}{FOOD_VIEW_PATH}"
        f"?manage_idx={manage_idx}&memo5={meal_date.isoformat()}"
    )


def _is_element(node: object) -> bool:
    return isinstance(node, _Element) and isinstance(node.tag, str)


def _text(el: _Element | None) -> str:
    if el is None or not _is_element(el):
        return ""
    return " ".join((el.text_content() or "").split())


def _split_dd_menus(dd: _Element) -> list[str]:
    """dd 내부를 <br> 단위로 분리합니다."""
    parts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        name = " ".join("".join(buf).split()).strip()
        buf.clear()
        if not name:
            return
        if _CLOSED_RE.search(name):
            return
        if name in {"-", "—", "–"}:
            return
        parts.append(name)

    if dd.text:
        buf.append(dd.text)
    for child in dd:
        if not _is_element(child):
            continue
        tag = child.tag.lower()
        if tag == "br":
            flush()
        else:
            buf.append(child.text_content() or "")
        if child.tail:
            buf.append(child.tail)
    flush()
    return parts


def parse_gknu_day_html(html: str, *, meal_date: date) -> list[dict[str, Any]]:
    """일별 view.do 응답을 meals DTO 조각으로 파싱합니다."""
    if not (html or "").strip():
        return []
    try:
        doc = lhtml.fromstring(html)
    except (ParserError, ValueError, TypeError):
        return []

    meals: list[dict[str, Any]] = []
    for dl in doc.xpath(".//dl"):
        if not _is_element(dl):
            continue
        label = ""
        dt = dl.find("dt")
        if dt is not None:
            span = dt.find(".//span")
            label = _text(span if span is not None else dt)
        meal_type = MEAL_LABEL_TO_TYPE.get(label)
        if not meal_type:
            continue
        menus: list[dict[str, Any]] = []
        for dd in dl.findall("dd"):
            for name in _split_dd_menus(dd):
                menus.append(
                    {
                        "cornerName": label,
                        "menuName": name,
                        "displayOrder": len(menus) + 1,
                    }
                )
        if not menus:
            continue
        meals.append(
            {
                "mealDate": meal_date.isoformat(),
                "mealType": meal_type,
                "menus": menus,
            }
        )
    meals.sort(
        key=lambda item: (
            item["mealDate"],
            {"BREAKFAST": 0, "LUNCH": 1, "DINNER": 2}.get(str(item["mealType"]), 99),
        )
    )
    return meals


def parse_gknu_western_html(html: str) -> list[dict[str, Any]]:
    """양식코너 고정 가격표 HTML에서 메뉴 목록을 추출합니다."""
    if not (html or "").strip():
        return []
    try:
        doc = lhtml.fromstring(html)
    except (ParserError, ValueError, TypeError):
        return []

    menus: list[dict[str, Any]] = []
    for table in doc.xpath(".//table"):
        if not _is_element(table):
            continue
        for tr in table.xpath(".//tr"):
            cells = tr.xpath("./td|./th")
            if len(cells) < 1:
                continue
            name = _text(cells[0])
            if not name or name in _HEADER_SKIP:
                continue
            if "대표메뉴" in name and "가격" in name:
                continue
            if _CLOSED_RE.search(name):
                continue
            menus.append(
                {
                    "cornerName": "양식코너",
                    "menuName": name,
                    "displayOrder": len(menus) + 1,
                }
            )
        if menus:
            break
    return menus


def _iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_gknu_daily_meals(
    *,
    cafeteria_name: str,
    source_url: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """경국대 식당별 소스를 조회해 표준 meals DTO를 반환합니다."""
    name = resolve_gknu_cafeteria_name(cafeteria_name, source_url)
    menu_idx = parse_menu_idx(source_url)
    assert menu_idx is not None  # resolve에서 이미 검증

    mondays = week_mondays_covering(start, end)
    if len(mondays) > MAX_WEEK_FETCHES:
        raise RuntimeError(
            f"경국대 식단 조회 기간은 최대 {MAX_WEEK_FETCHES}주까지 허용됩니다."
        )

    if menu_idx == WESTERN_MENU_IDX:
        html = fetch_html(source_url)
        template = parse_gknu_western_html(html)
        meals: list[dict[str, Any]] = []
        if not template:
            return meals
        for d in _iter_dates(start, end):
            # 양식코너 안내: 학기중 운영, 주말·공휴일 휴무 (공휴일 캘린더는 미적용)
            if d.weekday() >= 5:
                continue
            meals.append(
                {
                    "mealDate": d.isoformat(),
                    "mealType": "LUNCH",
                    "menus": [
                        {
                            "cornerName": item["cornerName"],
                            "menuName": item["menuName"],
                            "displayOrder": item["displayOrder"],
                        }
                        for item in template
                    ],
                }
            )
        return meals

    manage_idx = MENU_IDX_TO_MANAGE.get(menu_idx)
    if manage_idx is None:
        raise RuntimeError(f"경국대 manage_idx를 확인할 수 없습니다: menu_idx={menu_idx}")

    referer = urljoin(GKNU_ORIGIN, urlparse(source_url).path + "?" + urlparse(source_url).query)
    merged: list[dict[str, Any]] = []
    fetch_ok = 0
    last_fetch_error: BaseException | None = None
    started = time.monotonic()

    for d in _iter_dates(start, end):
        elapsed = time.monotonic() - started
        if elapsed >= DAY_FETCH_BUDGET_SECONDS:
            logger.warning(
                "gknu day fetch budget exhausted cafeteria=%s stopped_before=%s "
                "elapsed=%.1fs budget=%.1fs",
                name,
                d.isoformat(),
                elapsed,
                DAY_FETCH_BUDGET_SECONDS,
            )
            break
        url = food_view_url(manage_idx, d)
        try:
            html = fetch_html(
                url,
                timeout=DAY_FETCH_TIMEOUT,
                referer=referer or source_url,
            )
            fetch_ok += 1
        except (requests.exceptions.RequestException, OSError) as e:
            last_fetch_error = e
            logger.warning(
                "gknu day fetch failed cafeteria=%s date=%s: %s",
                name,
                d.isoformat(),
                e,
            )
            continue
        merged.extend(parse_gknu_day_html(html, meal_date=d))

    if fetch_ok == 0 and last_fetch_error is not None:
        raise last_fetch_error

    merged.sort(
        key=lambda item: (
            item["mealDate"],
            {"BREAKFAST": 0, "LUNCH": 1, "DINNER": 2}.get(str(item["mealType"]), 99),
        )
    )
    return merged
