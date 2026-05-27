"""Live service의 순수 비즈니스 로직 모듈."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
from datetime import date, datetime, timedelta
from io import StringIO
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pandas.errors import ParserError

from app.config.runtime import ALLOWED_ACCEPT_LANGUAGES, ServiceConfig
from app.domain.allergy.agent import analyze_menus_with_gemini, iter_menu_entries, results_to_dataframe
from app.domain.crawler.kumoh_menu import MENU_ITEM_DELIM, load_menus, normalize_kumoh_cafeteria_name, parse_table_from_html
from app.domain.crawler.push_menus import post_menu_ingest
from app.services.allergen_mapping import (
    format_mfds_labels_for_prompt,
    map_allergy_code,
    map_ingredient_code,
)
from utils.json_extract import extract_json_object
from user_features.i18n_summary import summarize_for_locale
from user_features.payloads import build_extended_menu_payload

DEFAULT_SOURCE_ALLOWLIST = {"www.kumoh.ac.kr", "kumoh.ac.kr"}
MEAL_TYPE_ORDER = {"BREAKFAST": 0, "LUNCH": 1, "DINNER": 2}
logger = logging.getLogger(__name__)

SPICY_LEVEL_MIN = 0
SPICY_LEVEL_MAX = 5


def parse_spicy_level(raw: Any) -> int | None:
    """모델 spicyLevel → 0~5. 미추정·파싱 실패 시 None, 0은 매운맛 없음(밥 등)."""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        n = int(float(raw))
    except (TypeError, ValueError):
        return None
    if n < SPICY_LEVEL_MIN or n > SPICY_LEVEL_MAX:
        return None
    return n


def clamp_spicy_level(raw: Any) -> int | None:
    """parse_spicy_level 별칭 (하위 호환)."""
    return parse_spicy_level(raw)


class CrawlSourceUpstreamError(Exception):
    """외부 sourceUrl fetch/파싱 실패가 최종적으로 해소되지 않은 경우."""


def auth_headers(token: str | None, api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def next_run(now: datetime, *, weekday: int, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    if days_ahead == 0 and candidate <= now:
        days_ahead = 7
    return candidate + timedelta(days=days_ahead)


def run_weekly_crawl_once(cfg: ServiceConfig, client: genai.Client | None) -> dict[str, Any]:
    if not cfg.spring_menus_url:
        raise RuntimeError("SPRING_MENUS_URL is required for weekly crawl forwarding")
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is required for weekly analysis")

    menus = load_menus()
    if not menus:
        raise RuntimeError("크롤링 결과가 비었습니다.")
    entries = iter_menu_entries(menus)
    if not entries:
        raise RuntimeError("분석할 메뉴 셀이 없습니다.")

    analysis_results = analyze_menus_with_gemini(
        client,
        cfg.weekly_menu_model,
        entries,
        batch_size=cfg.weekly_batch_size,
        sleep_between_batches_sec=cfg.weekly_sleep_seconds,
    )
    analysis_df = results_to_dataframe(analysis_results)
    i18n_rows = analysis_df.to_dict(orient="records")
    i18n_summary = summarize_for_locale(client, cfg.weekly_menu_model, i18n_rows, cfg.i18n_locale)

    payload = build_extended_menu_payload(
        menus,
        source="https://www.kumoh.ac.kr",
        analysis_df=analysis_df,
        i18n_summary=i18n_summary,
    )
    res = post_menu_ingest(
        cfg.spring_menus_url,
        payload,
        bearer_token=cfg.spring_api_token,
        api_key=cfg.spring_api_key,
        timeout=60.0,
    )
    if not res.ok:
        body = (res.text or "").strip()
        raise RuntimeError(f"메뉴 전송 실패 HTTP {res.status_code}: {body[:500]}")

    return {
        "status": "ok",
        "restaurants": len(payload.get("data", {}).get("restaurants", [])),
        "analysisRows": len(analysis_df),
        "i18nLocale": cfg.i18n_locale,
    }


def analyze_food_text(client: genai.Client | None, model_name: str, name: str) -> dict[str, Any]:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    mfds_labels = format_mfds_labels_for_prompt()
    prompt = f"""음식 이름: {name}

한국 식품의약품안전처 알레르기 유발물질 표시 대상 기준으로 분석합니다.

다음 JSON 객체 하나만 출력:
{{
  "foodNameKo": "음식 이름(한국어)",
  "ingredientsKo": ["주요 재료를 빠짐없이 한국어로. 고기·해산물·채소·양념·부재료 포함"],
  "allergensKo": [{{"name": "표준 알레르기명(아래 목록 중 하나)", "reason": "함유·가능 근거"}}],
  "spicyLevel": 2
}}

규칙:
- ingredientsKo: 메뉴에 들어갈 수 있는 주재료를 모두 나열(알레르기 유발 재료도 포함).
- allergensKo.name: 아래 표준명 문자열 그대로만 사용. 목록에 없는 이름(생선, 어류, 콩 등) 금지.
  표준명: {mfds_labels}
  예: 계란→난류, 콩·두부·간장→대두, 밀가루→밀, 치킨→닭고기
- 확실하지 않은 알레르기는 allergensKo에 넣지 말 것.
- spicyLevel: 매운맛 0~5 정수. 0=매운맛 없음(흰밥·미지근한 음식), 1=약함~5=아주 매움. 판단 불가하면 키 생략.
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("모델 응답 JSON이 객체 형태가 아닙니다.")
    raw_spicy = parsed.get("spicyLevel")
    if raw_spicy is None:
        raw_spicy = parsed.get("spicy_level")
    parsed["spicyLevel"] = parse_spicy_level(raw_spicy)
    return parsed


def identify_food_from_image(
    client: genai.Client | None,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = """이미지의 음식 이름만 식별하세요. JSON 객체 하나만 출력:
{"foodNameKo":"...", "confidence": 0.0~1.0}
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("모델 응답 JSON이 객체 형태가 아닙니다.")
    return parsed


def extract_menu_text_from_image(
    client: genai.Client | None,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = """메뉴판 이미지에서 메뉴 텍스트를 OCR 관점으로 읽어주세요.
JSON 객체 하나만 출력:
{
  "rawText": "메뉴판에서 읽은 전체 텍스트",
  "menuNames": ["중복 제거된 메뉴명", "메뉴명2"]
}
규칙:
- menuNames는 실제 음식/메뉴명만 포함
- 가격, 날짜, 번호, 안내문구 제외
- 같은 메뉴 중복 제거
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 OCR 응답이 비어 있습니다.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini OCR response: %s", raw)
        raise RuntimeError(f"모델 OCR 응답이 유효한 JSON이 아닙니다: {e}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("모델 OCR 응답 JSON이 객체 형태가 아닙니다.")

    raw_text = parsed.get("rawText")
    if not isinstance(raw_text, str):
        raw_text = ""
    menu_names_raw = parsed.get("menuNames")
    if not isinstance(menu_names_raw, list):
        menu_names_raw = []
    menu_names: list[str] = []
    dedup: set[str] = set()
    for entry in menu_names_raw:
        if not isinstance(entry, str):
            continue
        normalized = entry.strip()
        if not normalized or normalized in dedup:
            continue
        dedup.add(normalized)
        menu_names.append(normalized)
    return {"rawText": raw_text.strip(), "menuNames": menu_names}


def post_json(*, url: str, payload: dict[str, Any], token: str | None, api_key: str | None) -> requests.Response:
    return requests.post(
        url,
        json=payload,
        headers=auth_headers(token, api_key),
        timeout=60.0,
    )


def v1_success(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}


def v1_error(code: str, msg: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "code": code, "msg": msg},
    )


def validate_accept_language(lang: str | None) -> None:
    if not lang:
        return
    normalize_request_language(lang)


def normalize_request_language(lang: str | None) -> str:
    """요청 언어 코드를 허용 목록의 표준 코드(ko, en, zh-CN, vi, ja)로 정규화합니다."""
    if not lang or not str(lang).strip():
        raise ValueError("language는 필수입니다. 허용: ko, en, zh-CN, vi, ja")
    first = str(lang).split(",", 1)[0].strip()
    normalized = first.split(";", 1)[0].strip()
    if normalized in ALLOWED_ACCEPT_LANGUAGES:
        return normalized
    lowered = normalized.lower()
    if lowered.startswith("zh-cn") or lowered == "zh":
        return "zh-CN"
    base_lang = lowered.split("-", 1)[0]
    if base_lang in {"ko", "en", "vi", "ja"}:
        return base_lang
    raise ValueError(
        f"지원하지 않는 language: {normalized}. "
        "허용: ko, en, zh-CN, vi, ja"
    )


def infer_meal_type(column_name: str) -> str:
    s = column_name.upper()
    if "조식" in column_name or "BREAKFAST" in s:
        return "BREAKFAST"
    if "석식" in column_name or "DINNER" in s:
        return "DINNER"
    return "LUNCH"


_TIME_RANGE_RE = re.compile(r"^\d{1,2}:\d{2}\s*[~\-]\s*\d{1,2}:\d{2}$")
_META_BRACKET_RE = re.compile(r"^\[.*\]$")


def _is_menu_noise(line: str) -> bool:
    """시간 범위, 대괄호 메타정보, 별표 안내문 등 메뉴명이 아닌 항목 판별."""
    if _TIME_RANGE_RE.match(line):
        return True
    if _META_BRACKET_RE.match(line):
        return True
    if line.startswith("*"):
        return True
    return False


_KNOWN_CORNERS = frozenset({"조식", "중식", "석식", "일품요리"})

# 분식당 HTML에 '라면류'·'돈가스류'로만 올라오는 경우 개별 메뉴명으로 펼칩니다.
_BUNSIK_RAMEN_MENUS = ("떡만두라면", "얼큰라면", "치즈라면", "라면", "공깃밥")
_BUNSIK_PORK_CUTLET_MENUS = ("왕돈가스", "고구마돈가스", "치즈돈가스")


def _expand_bunsik_category_tokens(menu_items: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in menu_items:
        s = item.strip()
        if s == "라면류":
            expanded.extend(_BUNSIK_RAMEN_MENUS)
        elif s == "돈가스류":
            expanded.extend(_BUNSIK_PORK_CUTLET_MENUS)
        else:
            expanded.append(item)
    return expanded


def parse_menu_cell(cell_text: str, fallback_corner: str) -> tuple[str, str, list[str]]:
    """셀 텍스트를 파싱하여 (cornerName, mealType, [menuName, ...])을 반환합니다.

    구분자(|||)를 기준으로 항목을 분리하며,
    필터링 후 유효 항목이 하나만 남고 그것이 알려진 코너명이 아닌 경우
    메뉴명으로 취급합니다.
    """
    fallback_corner = normalize_kumoh_cafeteria_name(fallback_corner)
    has_delimiters = MENU_ITEM_DELIM in cell_text
    items = [s.strip() for s in cell_text.split(MENU_ITEM_DELIM) if s.strip()]

    corner_name = ""
    menu_items: list[str] = []

    for item in items:
        if not item or item.lower() == "nan":
            continue
        if "운영 없음" in item:
            continue
        if _is_menu_noise(item):
            continue
        if not corner_name:
            corner_name = item
            continue
        menu_items.append(item)

    if not menu_items and corner_name:
        if not has_delimiters or corner_name not in _KNOWN_CORNERS:
            menu_items = [corner_name]
            corner_name = fallback_corner

    if not corner_name:
        corner_name = fallback_corner

    meal_type = infer_meal_type(corner_name)
    return corner_name, meal_type, menu_items


def sanitize_url_for_log(source_url: str) -> str:
    parsed = urlparse(source_url)
    host = parsed.hostname or ""
    path = parsed.path or "/"
    return f"{parsed.scheme}://{host}{path}"


def extract_date_from_column(column_name: str, start: date, end: date) -> date | None:
    match = re.search(r"(\d{1,2})\.(\d{1,2})", column_name)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    candidate_years = [start.year]
    if end.year != start.year:
        candidate_years.append(end.year)

    for year in candidate_years:
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if start <= candidate <= end:
            return candidate
    return None


def build_daily_meals(*, cafeteria_name: str, table: Any, start: date, end: date) -> list[dict[str, Any]]:
    cafeteria_name = normalize_kumoh_cafeteria_name(cafeteria_name)
    meals: list[dict[str, Any]] = []
    for column in table.columns:
        meal_date = extract_date_from_column(str(column), start, end)
        if meal_date is None or not (start <= meal_date <= end):
            continue

        meals_by_type: dict[str, list[dict[str, Any]]] = {}

        for _, row in table.iterrows():
            raw = row[column]
            if raw is None:
                continue
            cell_text = str(raw).strip()
            if not cell_text or cell_text.lower() == "nan":
                continue

            corner_name, meal_type, menu_items = parse_menu_cell(cell_text, cafeteria_name)
            if not menu_items:
                continue
            if cafeteria_name == "분식당":
                menu_items = _expand_bunsik_category_tokens(menu_items)
                if not menu_items:
                    continue

            if meal_type not in meals_by_type:
                meals_by_type[meal_type] = []
            for item in menu_items:
                meals_by_type[meal_type].append(
                    {"cornerName": corner_name, "menuName": item}
                )

        for meal_type, menu_list in meals_by_type.items():
            for i, m in enumerate(menu_list, 1):
                m["displayOrder"] = i
            meals.append(
                {
                    "mealDate": meal_date.isoformat(),
                    "mealType": meal_type,
                    "menus": menu_list,
                }
            )

    meals.sort(
        key=lambda item: (
            item["mealDate"],
            MEAL_TYPE_ORDER.get(str(item["mealType"]), 99),
        )
    )
    return meals


def load_menu_table_for_source(*, cafeteria_name: str, source_url: str) -> pd.DataFrame:
    cafeteria_name = normalize_kumoh_cafeteria_name(cafeteria_name)
    _validate_source_url(source_url)
    source_fetch_error: BaseException | None = None

    try:
        response = requests.get(source_url, timeout=15, allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise requests.exceptions.RequestException("redirect is not allowed for source_url")
        response.raise_for_status()
        response.encoding = "utf-8"
        table = parse_table_from_html(response.text)
        if table is not None:
            return table
    except (
        requests.exceptions.RequestException,
        ParserError,
        ValueError,
        UnicodeError,
        OSError,
    ) as e:
        source_fetch_error = e
        logger.warning(
            "sourceUrl fetch/parse failed (source=%s): %s",
            sanitize_url_for_log(source_url),
            e,
            exc_info=True,
        )

    fallback_menus = load_menus()
    table = fallback_menus.get(cafeteria_name)
    if table is None:
        if source_fetch_error is not None:
            raise CrawlSourceUpstreamError(
                "sourceUrl fetch/parse failed and fallback cafeteria data was unavailable."
            ) from source_fetch_error
        raise RuntimeError(
            "sourceUrl에서 식단표 파싱에 실패했고, 등록된 식당명 기반 폴백도 실패했습니다."
        )
    return table


def _validate_source_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme.lower() != "https":
        raise RuntimeError("sourceUrl은 https만 허용됩니다.")
    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError("sourceUrl hostname이 비어 있습니다.")
    raw_allowlist = os.environ.get("CRAWL_SOURCE_ALLOWLIST", "").strip()
    if raw_allowlist:
        allowlist = {host.strip().lower() for host in raw_allowlist.split(",") if host.strip()}
    else:
        allowlist = set(DEFAULT_SOURCE_ALLOWLIST)
    normalized_host = hostname.lower()
    if normalized_host not in allowlist:
        raise RuntimeError(f"허용되지 않은 sourceUrl host입니다: {hostname}")
    try:
        infos = socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as e:
        raise RuntimeError(f"sourceUrl DNS 조회 실패: {e}") from e
    for info in infos:
        ip_text = info[4][0]
        ip_obj = ip_address(ip_text)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise RuntimeError("sourceUrl이 사설/내부/예약 IP로 해석되어 차단되었습니다.")


def describe_food_with_gemini(
    client: genai.Client | None,
    model_name: str,
    food_name: str,
) -> str:
    """음식명을 받아 한국어로 음식 설명을 생성합니다."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = f"""음식 이름: {food_name}

이 음식을 처음 접하는 사람에게 설명하는 **한국어** 문장 2~4개를 작성하세요.
무엇인지, 대표 재료, 맛·식감을 자연스럽게 설명합니다.
알레르기 경고나 분류 코드는 넣지 마세요.

JSON 객체 하나만 출력:
{{
  "description": "한국어 설명..."
}}
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    description: str | None = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            value = parsed.get("description") or parsed.get("descriptionKo")
            if isinstance(value, str) and value.strip():
                description = value.strip()
    except json.JSONDecodeError:
        pass
    if not description:
        fallback = _parse_json_field_fallback(raw, "description")
        if fallback:
            return fallback.strip()
        raise RuntimeError("모델 응답에 description이 없습니다.")
    return description


def translate_text_list_with_gemini(
    client: genai.Client | None,
    model_name: str,
    source_lang: str,
    target_lang: str,
    texts: list[str],
) -> list[str]:
    """문자열 목록을 한 번에 번역합니다. 입력 순서와 동일한 길이의 목록을 반환합니다."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if len(cleaned) != len(texts):
        raise RuntimeError("text 목록에 빈 문자열이 포함되어 있습니다.")
    if not cleaned:
        raise RuntimeError("text 목록이 비어 있습니다.")

    # 비용/토큰 최적화: 중복 텍스트는 한 번만 번역한 뒤 원래 순서로 복원합니다.
    unique_cleaned = list(dict.fromkeys(cleaned))

    numbered = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(unique_cleaned))
    prompt = f"""Translate each line from {source_lang} to {target_lang}.
Preserve the same count and order as the input. Each item is a food ingredient or short label.

Return ONE JSON object only:
{{
  "translatedTexts": ["...", "..."]
}}

Input ({len(unique_cleaned)} items):
{numbered}
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 번역 응답이 비어 있습니다.")

    translated_unique: list[str] | None = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            candidate = parsed.get("translatedTexts")
            if isinstance(candidate, list):
                if all(isinstance(x, str) for x in candidate):
                    translated_unique = [x.strip() for x in candidate]
        elif isinstance(parsed, list):
            if all(isinstance(x, str) for x in parsed):
                translated_unique = [x.strip() for x in parsed]
    except json.JSONDecodeError:
        pass

    if translated_unique is None or len(translated_unique) != len(unique_cleaned):
        raise RuntimeError(
            f"모델 번역 응답 개수가 요청과 일치하지 않습니다 (요청 {len(unique_cleaned)}개)."
        )
    if any(not item for item in translated_unique):
        raise RuntimeError("모델 번역 응답에 빈 문자열이 포함되어 있습니다.")

    mapping = dict(zip(unique_cleaned, translated_unique))
    return [mapping[item] for item in cleaned]


def translate_text_with_gemini(
    client: genai.Client | None,
    model_name: str,
    source_lang: str,
    target_lang: str,
    text: str,
) -> str:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = f"""Translate text from {source_lang} to {target_lang}.
Return one JSON object only:
{{
  "translatedText": "..."
}}
Input text:
{text}
"""
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        raise RuntimeError("모델 번역 응답이 비어 있습니다.")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("모델 번역 응답 JSON이 객체 형태가 아닙니다.")
    translated = parsed.get("translatedText")
    if not isinstance(translated, str) or not translated.strip():
        raise RuntimeError("모델 번역 응답 형식이 올바르지 않습니다.")
    return translated.strip()


def _parse_json_field_fallback(raw: str, field: str) -> str | None:
    """잘린 JSON 응답에서 문자열 필드 값을 최대한 복구합니다."""
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        return match.group(1).strip()
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]+)', raw)
    return match.group(1).strip() if match else None


def pronounce_food_name_with_gemini(
    client: genai.Client | None,
    model_name: str,
    target_lang: str,
    korean_name: str,
) -> str | None:
    """한국어 음식명의 발음 표기만 반환합니다."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    name = (korean_name or "").strip()
    if not name:
        return None
    prompt = f"""Return one JSON object only:
{{
  "pronunciation": "pronunciation of the Korean dish name for speakers of {target_lang}"
}}
Korean dish name: {name}
Do not translate the meaning. Only pronunciation/romanization."""
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=128,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = extract_json_object(raw)
        except ValueError:
            recovered = _parse_json_field_fallback(raw, "pronunciation")
            return recovered
    if not isinstance(parsed, dict):
        return _parse_json_field_fallback(raw, "pronunciation")
    value = parsed.get("pronunciation")
    return value.strip() if isinstance(value, str) and value.strip() else None


def localize_food_name_with_gemini(
    client: genai.Client | None,
    model_name: str,
    target_lang: str,
    korean_name: str,
) -> dict[str, str | None]:
    """한국어 음식명의 번역(의미)과 발음(표기)을 분리해 반환합니다."""
    name = (korean_name or "").strip()
    if not name:
        return {"translation": None, "pronunciation": None}
    if target_lang == "ko":
        return {"translation": name, "pronunciation": name}
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")

    translation = translate_text_with_gemini(client, model_name, "ko", target_lang, name)
    pronunciation: str | None = None
    try:
        pronunciation = pronounce_food_name_with_gemini(client, model_name, target_lang, name)
    except Exception as exc:
        logger.warning("pronunciation generation failed for %s (%s): %s", name, target_lang, exc)
    if not pronunciation:
        pronunciation = translation
    return {"translation": translation, "pronunciation": pronunciation}


__all__ = [
    "CrawlSourceUpstreamError",
    "auth_headers",
    "analyze_food_text",
    "build_daily_meals",
    "extract_menu_text_from_image",
    "extract_date_from_column",
    "identify_food_from_image",
    "infer_meal_type",
    "load_menu_table_for_source",
    "map_allergy_code",
    "map_ingredient_code",
    "next_run",
    "parse_menu_cell",
    "post_json",
    "run_weekly_crawl_once",
    "sanitize_url_for_log",
    "describe_food_with_gemini",
    "translate_text_list_with_gemini",
    "translate_text_with_gemini",
    "localize_food_name_with_gemini",
    "pronounce_food_name_with_gemini",
    "normalize_request_language",
    "validate_accept_language",
    "v1_error",
    "v1_success",
]
