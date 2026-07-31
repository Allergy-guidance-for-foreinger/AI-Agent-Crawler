"""경북대 생협(coop.knu.ac.kr) 식당 식단 HTML 파서."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from lxml import html as lhtml
from lxml.etree import _Element

logger = logging.getLogger(__name__)

KNU_HOSTS = frozenset({"coop.knu.ac.kr", "www.coop.knu.ac.kr"})

# shop_sqno → 식당명 (요청된 5개)
SHOP_NAMES: dict[int, str] = {
    35: "정보센터식당",
    36: "복지관 교직원식당",
    37: "카페테리아 첨성",
    46: "GP감꽃식당",
    85: "공학관교직원식당(외부업체)",
}

SHOP_BY_NAME: dict[str, int] = {name: sqno for sqno, name in SHOP_NAMES.items()}

_DATE_IN_HEADER_RE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")
_TIME_RANGE_RE = re.compile(
    r"\(\s*\d{1,2}\s*:\s*\d{2}\s*~\s*\d{1,2}\s*:\s*\d{2}"
    r"(?:\s*,\s*\d{1,2}\s*:\s*\d{2}\s*~\s*\d{1,2}\s*:\s*\d{2})*\s*\)"
)
_BARE_TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}\s*:\s*\d{2}\s*~\s*\d{1,2}\s*:\s*\d{2}\b"
)
_PRICE_RE = re.compile(r"[￦₩]\s*[\d,]+")
_OPERATING_RE = re.compile(r"운영\s*시간")


def is_knu_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    return hostname.lower() in KNU_HOSTS


def normalize_knu_cafeteria_name(name: str) -> str:
    return (name or "").strip()


def parse_shop_sqno(source_url: str) -> int | None:
    parsed = urlparse(source_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    raw = qs.get("shop_sqno", "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def resolve_knu_cafeteria_name(cafeteria_name: str, source_url: str) -> str:
    """요청 식당명을 우선하고, 비어 있으면 URL shop_sqno로 보완합니다."""
    name = normalize_knu_cafeteria_name(cafeteria_name)
    if name:
        return name
    sqno = parse_shop_sqno(source_url)
    if sqno is not None and sqno in SHOP_NAMES:
        return SHOP_NAMES[sqno]
    raise RuntimeError("경북대 식당명을 확인할 수 없습니다.")


def with_sel_date(source_url: str, sel_date: date) -> str:
    parsed = urlparse(source_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["selDate"] = sel_date.isoformat()
    return urlunparse(parsed._replace(query=urlencode(qs)))


def week_mondays_covering(start: date, end: date) -> list[date]:
    if start > end:
        return []
    monday = start - timedelta(days=start.weekday())
    out: list[date] = []
    cur = monday
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def fetch_html(url: str, *, timeout: float = 15.0) -> str:
    res = requests.get(
        url,
        timeout=timeout,
        allow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Agent-Crawler/1.0)"},
    )
    if 300 <= res.status_code < 400:
        raise requests.exceptions.RequestException("redirect is not allowed for source_url")
    res.raise_for_status()
    res.encoding = res.apparent_encoding or "utf-8"
    return res.text


def _is_element(node: object) -> bool:
    return isinstance(node, _Element) and isinstance(node.tag, str)


def _text(el: _Element | None) -> str:
    if el is None or not _is_element(el):
        return ""
    return " ".join((el.text_content() or "").split())


def _parse_header_dates(doc: _Element, start: date, end: date) -> list[date | None]:
    """주간 헤더 테이블에서 월~토 날짜 목록을 추출합니다."""
    header_tables = doc.xpath("//table[contains(@class,'tstyle_me')]")
    if not header_tables:
        header_tables = doc.xpath("//table[.//th[contains(., '월')]]")
    if not header_tables:
        return []

    ths = header_tables[0].xpath(".//th")
    dates: list[date | None] = []
    candidate_years = [start.year] + ([end.year] if end.year != start.year else [])
    for th in ths:
        label = _text(th)
        if "분류" in label or "주간" in label:
            continue
        match = _DATE_IN_HEADER_RE.search(label)
        if not match:
            dates.append(None)
            continue
        month, day = int(match.group(1)), int(match.group(2))
        resolved: date | None = None
        for year in candidate_years:
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            # 주간 헤더는 요청 기간 밖(주 전체)일 수 있어 연도만 맞으면 채택
            if resolved is None:
                resolved = candidate
            if start <= candidate <= end or abs((candidate - start).days) <= 7:
                resolved = candidate
                break
        dates.append(resolved)
    return dates


def _infer_meal_type(week_table: _Element, index: int) -> str:
    text = _text(week_table)[:40]
    if text.startswith("석식") or "석식" in text[:10]:
        return "DINNER"
    if text.startswith("조식") or "조식" in text[:10]:
        return "BREAKFAST"
    if text.startswith("중식") or "중식" in text[:10]:
        return "LUNCH"
    # 이미지 alt/파일명 보조
    for alt in week_table.xpath(".//img/@alt"):
        if "석식" in alt:
            return "DINNER"
        if "조식" in alt:
            return "BREAKFAST"
        if "중식" in alt:
            return "LUNCH"
    for src in week_table.xpath(".//img/@src"):
        lower = (src or "").lower()
        if "suk" in lower or "dinner" in lower:
            return "DINNER"
        if "jo" in lower or "break" in lower:
            return "BREAKFAST"
    return "DINNER" if index >= 1 else "LUNCH"


def _clean_menu_name(raw: str) -> str:
    text = raw or ""
    text = _PRICE_RE.sub(" ", text)
    text = _TIME_RANGE_RE.sub(" ", text)
    text = _BARE_TIME_RANGE_RE.sub(" ", text)
    text = _OPERATING_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\n\r/-")
    # 선행 코너 토큰 중복 제거(정식정식 → 정식)
    text = re.sub(r"^(정식|특식)(\s*\1)+", r"\1", text)
    return text.strip()


_CORNER_ONLY_NAMES = frozenset({"정식", "특식", "중식", "석식", "조식"})


def _body_text_from_li(li: _Element) -> str:
    body_parts: list[str] = []
    for p in li.xpath("./p"):
        p_text = _text(p)
        if not p_text or "￦" in p_text or "₩" in p_text:
            continue
        body_parts.append(p_text)
    return _clean_menu_name(" ".join(body_parts))


def _menu_name_from_li(li: _Element) -> str:
    """li 하나(가격 단위)에서 menuName을 만듭니다.

    제목(직접 텍스트/<br>)이 있으면 우선하고, 없으면 본문 p를 사용합니다.
    제목이 '정식'/'특식'처럼 코너명만이면 본문 메뉴를 사용합니다.
    """
    title_parts: list[str] = []
    if li.text and li.text.strip():
        title_parts.append(li.text.strip())
    for child in list(li):
        if not _is_element(child):
            continue
        if child.tag == "br":
            if child.tail and child.tail.strip():
                title_parts.append(child.tail.strip())
            continue
        if child.tag == "p":
            break
        title_parts.append(_text(child))
        if child.tail and child.tail.strip():
            title_parts.append(child.tail.strip())

    title = _clean_menu_name(" ".join(title_parts))
    body = _body_text_from_li(li)
    if not title:
        return body
    if title in _CORNER_ONLY_NAMES:
        return body or title
    return title


def _corner_from_cell(td: _Element, fallback: str) -> str:
    for btn in td.xpath(".//div[contains(@class,'button_m')]"):
        label = _text(btn)
        if label:
            return label
    return fallback


def parse_knu_week_html(
    html: str,
    *,
    cafeteria_name: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """단일 주 HTML → meals DTO 리스트(요청 기간으로 필터)."""
    doc = lhtml.fromstring(html)
    header_dates = _parse_header_dates(doc, start, end)
    if not header_dates:
        logger.warning("knu header dates missing cafeteria=%s", cafeteria_name)
        return []

    meals: list[dict[str, Any]] = []
    week_tables = doc.xpath("//div[contains(@class,'week_table')]")
    for table_idx, week_table in enumerate(week_tables):
        meal_type = _infer_meal_type(week_table, table_idx)
        data_rows = [
            tr
            for tr in week_table.xpath(".//tr")
            if len(tr.xpath("./td")) >= 1
        ]
        if not data_rows:
            continue
        # 보통 데이터 행 1개, 셀 = 월~토
        cells = data_rows[0].xpath("./td")
        for col_idx, td in enumerate(cells):
            if col_idx >= len(header_dates):
                break
            meal_date = header_dates[col_idx]
            if meal_date is None or not (start <= meal_date <= end):
                continue
            corner = _corner_from_cell(td, cafeteria_name)
            menus: list[dict[str, Any]] = []
            items = td.xpath(".//ul[contains(@class,'menu_im')]/li")
            if not items:
                # li 구조가 없으면 셀 전체에서 가격 단위 분리
                cell_text = _text(td)
                if not cell_text or cell_text.lower() == "nan":
                    continue
                chunks = re.split(r"(?<=\d)\s*(?=(?:특식|정식|라면|우동))", cell_text)
                if len(chunks) <= 1:
                    chunks = _PRICE_RE.split(cell_text)
                for chunk in chunks:
                    name = _clean_menu_name(chunk)
                    if not name:
                        continue
                    menus.append(
                        {
                            "cornerName": corner,
                            "menuName": name,
                            "displayOrder": len(menus) + 1,
                        }
                    )
            else:
                for li in items:
                    name = _menu_name_from_li(li)
                    if not name:
                        continue
                    menus.append(
                        {
                            "cornerName": corner,
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


def build_knu_daily_meals(
    *,
    cafeteria_name: str,
    source_url: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """기간을 덮는 주차별로 fetch 후 meals를 병합합니다."""
    name = resolve_knu_cafeteria_name(cafeteria_name, source_url)
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for monday in week_mondays_covering(start, end):
        week_url = with_sel_date(source_url, monday)
        html = fetch_html(week_url)
        week_meals = parse_knu_week_html(
            html,
            cafeteria_name=name,
            start=start,
            end=end,
        )
        for meal in week_meals:
            key = (meal["mealDate"], meal["mealType"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(meal)

    merged.sort(
        key=lambda item: (
            item["mealDate"],
            {"BREAKFAST": 0, "LUNCH": 1, "DINNER": 2}.get(str(item["mealType"]), 99),
        )
    )
    return merged
