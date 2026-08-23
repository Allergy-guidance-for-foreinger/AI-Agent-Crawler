"""경북대 생협(coop.knu.ac.kr) 식당 식단 HTML 파서."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from lxml import html as lhtml
from lxml.etree import ParserError, _Element

logger = logging.getLogger(__name__)

KNU_HOSTS = frozenset({"coop.knu.ac.kr", "www.coop.knu.ac.kr"})
# 식단 페이지 정규 경로. 루트 `/?shop_sqno=` 만으로는 week_table이 오지 않습니다.
KNU_MENU_PATH = "/sub03/sub01_01.html"

# shop_sqno → 식당명 (요청된 5개)
SHOP_NAMES: dict[int, str] = {
    35: "정보센터식당",
    36: "복지관 교직원식당",
    37: "카페테리아 첨성",
    46: "GP감꽃식당",
    85: "공학관교직원식당(외부업체)",
}

SHOP_BY_NAME: dict[str, int] = {name: sqno for sqno, name in SHOP_NAMES.items()}
MAX_WEEK_FETCHES = 5
# coop.knu.ac.kr 주간 페이지는 selDate부터 연속 6일(헤더 6칸)을 보여 줍니다.
# 헤더의 월~토 라벨은 요일명이 아니라 칸 순서 표시입니다.
KNU_PAGE_DAYS = 6

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


def normalize_knu_source_url(source_url: str) -> str:
    """경북대 sourceUrl을 식단 페이지 정규 경로로 맞춥니다.

    Spring 등에서 `https://coop.knu.ac.kr/?shop_sqno=35` 처럼 경로를 생략해도
    조회에 필요한 `/sub03/sub01_01.html` 로 보정합니다.
    """
    parsed = urlparse(source_url)
    if not is_knu_host(parsed.hostname):
        return source_url
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "shop_sqno" not in qs:
        return source_url
    path = parsed.path or ""
    if path.rstrip("/") == KNU_MENU_PATH.rstrip("/"):
        return source_url
    return urlunparse(
        parsed._replace(path=KNU_MENU_PATH, query=urlencode(qs), params="", fragment="")
    )


def resolve_knu_cafeteria_name(cafeteria_name: str, source_url: str) -> str:
    """shop_sqno 기준 식당명을 확정하고, 요청명이 있으면 일치 여부를 검증합니다."""
    sqno = parse_shop_sqno(source_url)
    if sqno is None or sqno not in SHOP_NAMES:
        raise RuntimeError("경북대 shop_sqno를 확인할 수 없거나 지원하지 않는 식당입니다.")
    expected = SHOP_NAMES[sqno]
    name = normalize_knu_cafeteria_name(cafeteria_name)
    if not name:
        return expected
    if name != expected:
        raise RuntimeError(
            f"경북대 식당명이 shop_sqno와 일치하지 않습니다: "
            f"cafeteriaName={name}, expected={expected}(shop_sqno={sqno})"
        )
    return expected


def with_sel_date(source_url: str, sel_date: date) -> str:
    normalized = normalize_knu_source_url(source_url)
    parsed = urlparse(normalized)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["selDate"] = sel_date.isoformat()
    return urlunparse(parsed._replace(query=urlencode(qs)))


def week_mondays_covering(start: date, end: date) -> list[date]:
    """요청 기간을 덮는 월요일 목록(호환용). 실제 fetch는 week_sel_dates_covering을 사용하세요."""
    if start > end:
        return []
    monday = start - timedelta(days=start.weekday())
    out: list[date] = []
    cur = monday
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def week_sel_dates_covering(start: date, end: date) -> list[date]:
    """요청 기간을 덮는 selDate 앵커 목록.

    경북대 페이지는 selDate를 첫 칸 날짜로 두고 이후 5일을 더해 총 6일을 표시합니다.
    월요일 고정이 아니므로, start부터 KNU_PAGE_DAYS 간격으로 조회합니다.
    """
    if start > end:
        return []
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=KNU_PAGE_DAYS)
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
    """주간 헤더 테이블에서 6칸 날짜 목록을 추출합니다.

    날짜가 없는 선행 th(분류/구분 등)는 라벨 문자열이 아니라 구조적으로 제거해
    데이터 셀 인덱스와 맞춥니다.
    """
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

    # 선행 비날짜 열 제거(분류/구분/주간 등 라벨 변경에도 정렬 유지)
    while dates and dates[0] is None:
        dates.pop(0)
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
    if not (html or "").strip():
        logger.warning("knu html empty cafeteria=%s", cafeteria_name)
        return []
    try:
        doc = lhtml.fromstring(html)
    except ParserError:
        logger.warning("knu html parse failed cafeteria=%s", cafeteria_name)
        return []

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
        if len(cells) != len(header_dates):
            logger.warning(
                "knu header/cell count mismatch cafeteria=%s headers=%s cells=%s",
                cafeteria_name,
                len(header_dates),
                len(cells),
            )
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
    """기간을 덮는 selDate 창별로 fetch 후 meals를 병합합니다."""
    source_url = normalize_knu_source_url(source_url)
    name = resolve_knu_cafeteria_name(cafeteria_name, source_url)
    sel_dates = week_sel_dates_covering(start, end)
    if len(sel_dates) > MAX_WEEK_FETCHES:
        raise RuntimeError(
            f"경북대 식단 조회 기간은 최대 {MAX_WEEK_FETCHES}주까지 허용됩니다."
        )

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    fetch_ok = 0
    last_fetch_error: BaseException | None = None

    for sel_date in sel_dates:
        week_url = with_sel_date(source_url, sel_date)
        try:
            html = fetch_html(week_url)
            fetch_ok += 1
        except (requests.exceptions.RequestException, OSError) as e:
            last_fetch_error = e
            logger.warning(
                "knu week fetch failed cafeteria=%s selDate=%s: %s",
                name,
                sel_date.isoformat(),
                e,
            )
            continue

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

    if fetch_ok == 0 and last_fetch_error is not None:
        raise last_fetch_error

    merged.sort(
        key=lambda item: (
            item["mealDate"],
            {"BREAKFAST": 0, "LUNCH": 1, "DINNER": 2}.get(str(item["mealType"]), 99),
        )
    )
    return merged
