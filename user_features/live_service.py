"""상시 구동 서비스: 주간 급식 크롤링 전송 + 실시간 이미지 분석 전송."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi import Body
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from google.genai import types
from pandas.errors import ParserError
from pydantic import BaseModel, Field

import repo_env
from crawler.kumoh_menu import load_menus
from crawler.push_menus import post_menu_ingest
from food_image.agent import analyze_food_image_bytes
from menu_allergy.agent import analyze_menus_with_gemini, iter_menu_entries, results_to_dataframe
from user_features.allergen_catalog import ALIAS_TO_CANONICAL
from user_features.i18n_summary import summarize_for_locale
from user_features.payloads import build_extended_menu_payload

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


WEEKDAY_TO_INDEX = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

API_V1_PREFIX = "/api/v1"
ALLOWED_ACCEPT_LANGUAGES = {"ko", "en", "zh-CN", "vi", "ja"}
CANONICAL_TO_INGREDIENT_CODE = {
    "난류": "EGG",
    "우유": "MILK",
    "메밀": "BUCKWHEAT",
    "땅콩": "PEANUT",
    "대두": "SOYBEAN",
    "밀": "WHEAT",
    "고등어": "MACKEREL",
    "게": "CRAB",
    "새우": "SHRIMP",
    "돼지고기": "PORK",
    "복숭아": "PEACH",
    "토마토": "TOMATO",
    "아황산류": "SULFITE",
    "호두": "WALNUT",
    "닭고기": "CHICKEN",
    "쇠고기": "BEEF",
    "오징어": "SQUID",
    "조개류": "SHELLFISH",
    "잣": "PINE_NUT",
}


class PythonMealCrawlRequest(BaseModel):
    schoolName: str = Field(..., min_length=1)
    cafeteriaName: str = Field(..., min_length=1)
    sourceUrl: str = Field(..., min_length=1)
    startDate: date
    endDate: date


class PythonMenuAnalysisTargetDto(BaseModel):
    menuId: int
    menuName: str = Field(..., min_length=1)


class PythonMenuAnalysisRequest(BaseModel):
    menus: list[PythonMenuAnalysisTargetDto] = Field(..., min_length=1)


class PythonMenuTranslationTargetDto(BaseModel):
    menuId: int
    menuName: str = Field(..., min_length=1)


class PythonMenuTranslationRequest(BaseModel):
    menus: list[PythonMenuTranslationTargetDto] = Field(..., min_length=1)
    targetLanguages: list[str] = Field(..., min_length=1)


class FreeTranslationRequest(BaseModel):
    sourceLang: str = Field(..., min_length=2)
    targetLang: str = Field(..., min_length=2)
    text: str = Field(..., min_length=1)


@dataclass
class ServiceConfig:
    spring_menus_url: str | None
    spring_image_analysis_url: str | None
    spring_image_identify_url: str | None
    spring_text_analysis_url: str | None
    spring_api_token: str | None
    spring_api_key: str | None
    gemini_model: str
    weekly_menu_model: str
    i18n_locale: str
    weekly_batch_size: int
    weekly_sleep_seconds: float
    enable_direct_image_analysis: bool
    timezone_name: str
    crawl_weekday: int
    crawl_hour: int
    crawl_minute: int


def _load_config() -> ServiceConfig:
    weekday_text = os.environ.get("WEEKLY_CRAWL_DAY", "mon").strip().lower()
    if weekday_text not in WEEKDAY_TO_INDEX:
        raise RuntimeError("WEEKLY_CRAWL_DAY must be one of mon,tue,wed,thu,fri,sat,sun")

    raw_hour = os.environ.get("WEEKLY_CRAWL_HOUR", "6")
    raw_minute = os.environ.get("WEEKLY_CRAWL_MINUTE", "0")
    try:
        hour = int(raw_hour)
        minute = int(raw_minute)
    except ValueError as e:
        raise RuntimeError(
            "WEEKLY_CRAWL_HOUR/WEEKLY_CRAWL_MINUTE must be integers "
            "(hour: 0..23, minute: 0..59)"
        ) from e
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise RuntimeError("WEEKLY_CRAWL_HOUR/MINUTE out of range")

    enable_direct_image_analysis = (
        os.environ.get("ENABLE_DIRECT_IMAGE_ANALYSIS", "false").strip().lower() == "true"
    )

    return ServiceConfig(
        spring_menus_url=os.environ.get("SPRING_MENUS_URL", "").strip() or None,
        spring_image_analysis_url=os.environ.get("SPRING_IMAGE_ANALYSIS_URL", "").strip() or None,
        spring_image_identify_url=os.environ.get("SPRING_IMAGE_IDENTIFY_URL", "").strip() or None,
        spring_text_analysis_url=os.environ.get("SPRING_TEXT_ANALYSIS_URL", "").strip() or None,
        spring_api_token=os.environ.get("SPRING_API_TOKEN", "").strip() or None,
        spring_api_key=os.environ.get("SPRING_API_KEY", "").strip() or None,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        weekly_menu_model=os.environ.get("WEEKLY_MENU_MODEL", "gemini-2.5-flash-lite"),
        i18n_locale=os.environ.get("I18N_LOCALE", "en"),
        weekly_batch_size=int(os.environ.get("WEEKLY_MENU_BATCH_SIZE", "4")),
        weekly_sleep_seconds=float(os.environ.get("WEEKLY_MENU_SLEEP_SECONDS", "21.0")),
        enable_direct_image_analysis=enable_direct_image_analysis,
        timezone_name=os.environ.get("SERVICE_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul",
        crawl_weekday=WEEKDAY_TO_INDEX[weekday_text],
        crawl_hour=hour,
        crawl_minute=minute,
    )


def _auth_headers(token: str | None, api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _next_run(now: datetime, *, weekday: int, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    if days_ahead == 0 and candidate <= now:
        days_ahead = 7
    return candidate + timedelta(days=days_ahead)


def _run_weekly_crawl_once(cfg: ServiceConfig) -> dict[str, Any]:
    if not cfg.spring_menus_url:
        raise RuntimeError("SPRING_MENUS_URL is required for weekly crawl forwarding")
    if CLIENT is None:
        raise RuntimeError("GEMINI_API_KEY is required for weekly analysis")

    menus = load_menus()
    if not menus:
        raise RuntimeError("크롤링 결과가 비었습니다.")

    entries = iter_menu_entries(menus)
    if not entries:
        raise RuntimeError("분석할 메뉴 셀이 없습니다.")

    analysis_results = analyze_menus_with_gemini(
        CLIENT,
        cfg.weekly_menu_model,
        entries,
        batch_size=cfg.weekly_batch_size,
        sleep_between_batches_sec=cfg.weekly_sleep_seconds,
    )
    analysis_df = results_to_dataframe(analysis_results)
    i18n_rows = analysis_df.to_dict(orient="records")
    i18n_summary = summarize_for_locale(CLIENT, cfg.weekly_menu_model, i18n_rows, cfg.i18n_locale)

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
        "restaurants": len(payload.get("restaurants", [])),
        "analysisRows": len(analysis_df),
        "i18nLocale": cfg.i18n_locale,
    }


def _analyze_food_text(name: str) -> dict[str, Any]:
    if CLIENT is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = f"""음식 이름: {name}

다음 JSON 객체 하나만 출력:
{{
  "foodNameKo": "음식 이름(한국어)",
  "ingredientsKo": ["주요 재료"],
  "allergensKo": [{{"name": "알레르기 유발 가능 식품", "reason": "근거"}}]
}}
"""
    resp = CLIENT.models.generate_content(
        model=CONFIG.gemini_model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
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


def _identify_food_from_image(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    if CLIENT is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = """이미지의 음식 이름만 식별하세요. JSON 객체 하나만 출력:
{"foodNameKo":"...", "confidence": 0.0~1.0}
"""
    resp = CLIENT.models.generate_content(
        model=CONFIG.gemini_model,
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


def _post_json(url: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        url,
        json=payload,
        headers=_auth_headers(CONFIG.spring_api_token, CONFIG.spring_api_key),
        timeout=60.0,
    )


def _v1_success(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}


def _v1_error(code: str, msg: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "code": code,
            "msg": msg,
        },
    )


def _validate_accept_language(lang: str | None) -> None:
    if not lang:
        return
    first = lang.split(",", 1)[0].strip()
    if not first:
        return
    normalized = first.split(";", 1)[0].strip()
    # Accept common locale variants for practical interoperability:
    # e.g., en-US -> en, ko-KR -> ko
    lowered = normalized.lower()
    if normalized in ALLOWED_ACCEPT_LANGUAGES:
        return
    if lowered.startswith("zh-cn"):
        return
    base_lang = lowered.split("-", 1)[0]
    if base_lang in {"ko", "en", "vi", "ja"}:
        return
    raise ValueError(
        f"지원하지 않는 Accept-Language: {normalized}. "
        "허용: ko, en, zh-CN, vi, ja"
    )


def _infer_meal_type(column_name: str) -> str:
    s = column_name.upper()
    if "조식" in column_name or "BREAKFAST" in s:
        return "BREAKFAST"
    if "석식" in column_name or "DINNER" in s:
        return "DINNER"
    return "LUNCH"


def _extract_date_from_column(column_name: str, year: int) -> date | None:
    match = re.search(r"(\d{1,2})\.(\d{1,2})", column_name)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _build_daily_meals(
    *,
    cafeteria_name: str,
    table: Any,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    meals: list[dict[str, Any]] = []
    for column in table.columns:
        meal_date = _extract_date_from_column(str(column), start.year)
        if meal_date is None or not (start <= meal_date <= end):
            continue
        menus: list[dict[str, Any]] = []
        display_order = 1
        first_column = table.columns[0] if len(table.columns) > 0 else None
        for _, row in table.iterrows():
            raw_menu = row[column]
            if raw_menu is None:
                continue
            menu_name = str(raw_menu).strip()
            if not menu_name or menu_name.lower() == "nan" or "운영 없음" in menu_name:
                continue
            corner_name = cafeteria_name
            if first_column is not None and first_column != column:
                first_col_text = str(row[first_column]).strip()
                if first_col_text and first_col_text.lower() != "nan":
                    corner_name = first_col_text
            menus.append(
                {
                    "cornerName": corner_name,
                    "displayOrder": display_order,
                    "menuName": menu_name,
                }
            )
            display_order += 1
        if menus:
            meals.append(
                {
                    "mealDate": meal_date.isoformat(),
                    "mealType": _infer_meal_type(str(column)),
                    "menus": menus,
                }
            )
    meals.sort(key=lambda item: (item["mealDate"], item["mealType"]))
    return meals


def _load_menu_table_for_source(
    *,
    cafeteria_name: str,
    source_url: str,
) -> pd.DataFrame:
    """요청 sourceUrl을 우선 사용해 식단 테이블을 로드한다.

    - sourceUrl에서 첫 번째 HTML 테이블 파싱
    - 실패 시 등록된 식당명(cafeteriaName) 기반 기본 크롤러로 폴백
    """
    try:
        response = requests.get(source_url, timeout=15)
        response.raise_for_status()
        response.encoding = "utf-8"
        tables = pd.read_html(StringIO(response.text))
        if tables:
            table = tables[0].copy()
            table.columns = [str(c).strip() for c in table.columns]
            table = table.replace(r"\s+", " ", regex=True)
            return table
    except (
        requests.exceptions.RequestException,
        ParserError,
        ValueError,
        UnicodeError,
        OSError,
    ):
        pass

    fallback_menus = load_menus()
    table = fallback_menus.get(cafeteria_name)
    if table is None:
        raise RuntimeError(
            "sourceUrl에서 식단표 파싱에 실패했고, 등록된 식당명 기반 폴백도 실패했습니다."
        )
    return table


def _map_ingredient_code(token: str) -> str | None:
    normalized = token.strip()
    if not normalized:
        return None
    direct = CANONICAL_TO_INGREDIENT_CODE.get(normalized)
    if direct:
        return direct

    alias_key = normalized.lower() if normalized.isascii() else normalized
    canonical = ALIAS_TO_CANONICAL.get(normalized) or ALIAS_TO_CANONICAL.get(alias_key)
    if canonical:
        return CANONICAL_TO_INGREDIENT_CODE.get(canonical)
    return None


def _translate_text_with_gemini(source_lang: str, target_lang: str, text: str) -> str:
    if CLIENT is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = f"""Translate text from {source_lang} to {target_lang}.
Return one JSON object only:
{{
  "translatedText": "..."
}}
Input text:
{text}
"""
    resp = CLIENT.models.generate_content(
        model=CONFIG.gemini_model,
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
    translated = parsed.get("translatedText")
    if not isinstance(translated, str) or not translated.strip():
        raise RuntimeError("모델 번역 응답 형식이 올바르지 않습니다.")
    return translated.strip()


repo_env.load_dotenv_from_repo_root()
CONFIG = _load_config()
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
CLIENT = genai.Client(api_key=API_KEY) if API_KEY else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async def _weekly_loop() -> None:
        tz = ZoneInfo(CONFIG.timezone_name)
        while True:
            now = datetime.now(tz)
            target = _next_run(
                now,
                weekday=CONFIG.crawl_weekday,
                hour=CONFIG.crawl_hour,
                minute=CONFIG.crawl_minute,
            )
            await asyncio.sleep(max((target - now).total_seconds(), 1))
            try:
                result = await asyncio.to_thread(_run_weekly_crawl_once, CONFIG)
                logger.info("weekly crawl forwarding succeeded: %s", result)
            except Exception:
                # 실패 시 프로세스를 죽이지 않고 다음 주기를 기다린다.
                logger.exception("weekly crawl forwarding failed")

    app.state.weekly_task = asyncio.create_task(_weekly_loop())
    try:
        yield
    finally:
        task = getattr(app.state, "weekly_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="AI-Agent-Crawler Live Service", lifespan=lifespan)
api_v1_router = APIRouter(prefix=API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith(API_V1_PREFIX):
        return _v1_error(
            "COM_002",
            "요청 데이터 변환 과정에서 오류가 발생했습니다.",
            status_code=400,
        )
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "weeklyCrawlConfigured": CONFIG.spring_menus_url is not None,
        "imageAnalysisConfigured": CONFIG.spring_image_analysis_url is not None,
        "imageIdentifyConfigured": CONFIG.spring_image_identify_url is not None,
        "textAnalysisConfigured": CONFIG.spring_text_analysis_url is not None,
        "directImageAnalysisEnabled": CONFIG.enable_direct_image_analysis,
        "timezone": CONFIG.timezone_name,
    }


@app.post("/crawl-and-forward")
def crawl_and_forward(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> dict[str, Any]:
    if CONFIG.spring_api_token and (
        credentials is None or credentials.credentials != CONFIG.spring_api_token
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return _run_weekly_crawl_once(CONFIG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/analyze-image-and-forward")
async def analyze_image_and_forward(
    image: UploadFile = File(...),
    user_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    if not CONFIG.enable_direct_image_analysis:
        raise HTTPException(
            status_code=403,
            detail=(
                "Direct image analysis is disabled. "
                "Set ENABLE_DIRECT_IMAGE_ANALYSIS=true to enable."
            ),
        )
    if CLIENT is None:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
    if not CONFIG.spring_image_analysis_url:
        raise HTTPException(status_code=500, detail="SPRING_IMAGE_ANALYSIS_URL is not set")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="이미지 파일이 너무 큽니다 (최대 10MB).")

    mime_type = image.content_type or "image/jpeg"
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 이미지 형식: {mime_type}")
    try:
        analysis = await asyncio.to_thread(
            analyze_food_image_bytes, CLIENT, CONFIG.gemini_model, image_bytes, mime_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 실패: {e}") from e

    payload = {
        "requestId": request_id,
        "userId": user_id,
        "capturedAt": datetime.now(ZoneInfo(CONFIG.timezone_name)).isoformat(),
        "analysis": analysis,
    }
    try:
        res = await asyncio.to_thread(
            _post_json,
            CONFIG.spring_image_analysis_url,
            payload,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Spring 전송 실패: {e}") from e

    if not res.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Spring 응답 오류 HTTP {res.status_code}: {(res.text or '')[:500]}",
        )
    return {"status": "ok", "forwardStatus": res.status_code, "analysis": analysis}


@app.post("/analyze-food-text-and-forward")
async def analyze_food_text_and_forward(
    food_name: str = Body(..., embed=True, description="분석할 음식 이름"),
) -> dict[str, Any]:
    if not CONFIG.spring_text_analysis_url:
        raise HTTPException(status_code=500, detail="SPRING_TEXT_ANALYSIS_URL is not set")
    if not food_name.strip():
        raise HTTPException(status_code=400, detail="food_name is empty")
    try:
        analysis = await asyncio.to_thread(_analyze_food_text, food_name.strip())
        payload = {
            "foodNameInput": food_name.strip(),
            "capturedAt": datetime.now(ZoneInfo(CONFIG.timezone_name)).isoformat(),
            "analysis": analysis,
        }
        res = await asyncio.to_thread(_post_json, CONFIG.spring_text_analysis_url, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 음식 분석 실패: {e}") from e

    if not res.ok:
        raise HTTPException(status_code=502, detail=f"Spring 응답 오류 HTTP {res.status_code}")
    return {"status": "ok", "forwardStatus": res.status_code, "analysis": analysis}


@app.post("/identify-image-and-forward")
async def identify_image_and_forward(
    image: UploadFile = File(...),
    request_id: str | None = None,
) -> dict[str, Any]:
    if not CONFIG.spring_image_identify_url:
        raise HTTPException(status_code=500, detail="SPRING_IMAGE_IDENTIFY_URL is not set")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="이미지 파일이 너무 큽니다 (최대 10MB).")
    mime_type = image.content_type or "image/jpeg"
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 이미지 형식: {mime_type}")

    try:
        identified = await asyncio.to_thread(_identify_food_from_image, image_bytes, mime_type)
        payload = {
            "requestId": request_id,
            "capturedAt": datetime.now(ZoneInfo(CONFIG.timezone_name)).isoformat(),
            "identified": identified,
        }
        res = await asyncio.to_thread(_post_json, CONFIG.spring_image_identify_url, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 음식 식별 실패: {e}") from e
    if not res.ok:
        raise HTTPException(status_code=502, detail=f"Spring 응답 오류 HTTP {res.status_code}")
    return {"status": "ok", "forwardStatus": res.status_code, "identified": identified}


@api_v1_router.post("/python/meals/crawl")
def crawl_meals_v1(payload: PythonMealCrawlRequest, request: Request):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    if payload.startDate > payload.endDate:
        return _v1_error("COM_001", "startDate는 endDate보다 이후일 수 없습니다.", status_code=400)

    try:
        table = _load_menu_table_for_source(
            cafeteria_name=payload.cafeteriaName,
            source_url=payload.sourceUrl,
        )
    except Exception:
        return _v1_error(
            "PYM_404",
            (
                "요청 식단을 찾을 수 없습니다. "
                f"(cafeteriaName={payload.cafeteriaName}, sourceUrl={payload.sourceUrl})"
            ),
            status_code=404,
        )

    meals = _build_daily_meals(
        cafeteria_name=payload.cafeteriaName,
        table=table,
        start=payload.startDate,
        end=payload.endDate,
    )
    return _v1_success(
        {
            "schoolName": payload.schoolName,
            "cafeteriaName": payload.cafeteriaName,
            "sourceUrl": payload.sourceUrl,
            "startDate": payload.startDate.isoformat(),
            "endDate": payload.endDate.isoformat(),
            "meals": meals,
        }
    )


@api_v1_router.post("/python/menus/analyze")
async def analyze_menus_v1(payload: PythonMenuAnalysisRequest, request: Request):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    if CLIENT is None:
        return _v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

    results: list[dict[str, Any]] = []
    for target in payload.menus:
        analyzed_at = datetime.now(ZoneInfo(CONFIG.timezone_name)).isoformat(timespec="seconds")
        try:
            analysis = await asyncio.to_thread(_analyze_food_text, target.menuName)
            ingredient_codes: list[dict[str, Any]] = []
            dedup: set[str] = set()

            for idx, ingredient in enumerate(analysis.get("ingredientsKo", [])):
                token = str(ingredient).strip()
                code = _map_ingredient_code(token)
                if not code or code in dedup:
                    continue
                dedup.add(code)
                ingredient_codes.append(
                    {
                        "ingredientCode": code,
                        "confidence": round(max(0.5, 0.95 - (idx * 0.07)), 2),
                    }
                )

            for allergen in analysis.get("allergensKo", []):
                if not isinstance(allergen, dict):
                    continue
                code = _map_ingredient_code(str(allergen.get("name", "")).strip())
                if not code or code in dedup:
                    continue
                dedup.add(code)
                ingredient_codes.append(
                    {
                        "ingredientCode": code,
                        "confidence": 0.8,
                    }
                )

            results.append(
                {
                    "menuId": target.menuId,
                    "menuName": target.menuName,
                    "status": "COMPLETED",
                    "reason": None,
                    "modelName": "gemini",
                    "modelVersion": CONFIG.gemini_model,
                    "analyzedAt": analyzed_at,
                    "ingredients": ingredient_codes,
                }
            )
        except Exception as e:
            results.append(
                {
                    "menuId": target.menuId,
                    "menuName": target.menuName,
                    "status": "FAILED",
                    "reason": str(e)[:300],
                    "modelName": "gemini",
                    "modelVersion": CONFIG.gemini_model,
                    "analyzedAt": analyzed_at,
                    "ingredients": [],
                }
            )

    return _v1_success({"results": results})


@api_v1_router.post("/python/menus/translate")
async def translate_menus_v1(payload: PythonMenuTranslationRequest, request: Request):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    if CLIENT is None:
        return _v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

    results: list[dict[str, Any]] = []
    for menu in payload.menus:
        translations: list[dict[str, str]] = []
        for lang in payload.targetLanguages:
            lang_code = lang.strip()
            if not lang_code:
                continue
            try:
                translated = await asyncio.to_thread(
                    _translate_text_with_gemini,
                    "ko",
                    lang_code,
                    menu.menuName,
                )
                translations.append(
                    {
                        "langCode": lang_code,
                        "translatedName": translated,
                    }
                )
            except Exception:
                continue
        results.append(
            {
                "menuId": menu.menuId,
                "sourceName": menu.menuName,
                "translations": translations,
            }
        )

    return _v1_success({"results": results})


@api_v1_router.post("/translations")
async def free_translation_v1(payload: FreeTranslationRequest, request: Request):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    try:
        translated = await asyncio.to_thread(
            _translate_text_with_gemini,
            payload.sourceLang,
            payload.targetLang,
            payload.text,
        )
    except Exception as e:
        return _v1_error("AI_002", f"번역 실패: {e}", status_code=500)

    return _v1_success(
        {
            "sourceLang": payload.sourceLang,
            "targetLang": payload.targetLang,
            "text": payload.text,
            "translatedText": translated,
        }
    )


@api_v1_router.post("/ai/menu-board/analyze")
async def analyze_menu_board_v1(
    request: Request,
    image: UploadFile = File(...),
    requestId: str | None = Form(default=None),
):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    if CLIENT is None:
        return _v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

    image_bytes = await image.read()
    if not image_bytes:
        return _v1_error("COM_001", "이미지 파일이 비어 있습니다.", status_code=400)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return _v1_error("COM_001", "이미지 파일이 너무 큽니다 (최대 10MB).", status_code=400)
    mime_type = image.content_type or "image/jpeg"
    if mime_type not in ALLOWED_MIME_TYPES:
        return _v1_error("COM_001", f"지원하지 않는 이미지 형식: {mime_type}", status_code=400)

    try:
        identified = await asyncio.to_thread(_identify_food_from_image, image_bytes, mime_type)
    except Exception as e:
        return _v1_error("AI_003", f"메뉴판 이미지 분석 실패: {e}", status_code=500)

    return _v1_success(
        {
            "requestId": requestId,
            "recognizedMenus": [
                {
                    "menuName": identified.get("foodNameKo"),
                    "confidence": identified.get("confidence"),
                }
            ],
        }
    )


@api_v1_router.post("/ai/food-images/analyze")
async def analyze_food_image_v1(
    request: Request,
    image: UploadFile = File(...),
    requestId: str | None = Form(default=None),
):
    try:
        _validate_accept_language(request.headers.get("Accept-Language"))
    except ValueError as e:
        return _v1_error("COM_001", str(e), status_code=400)
    if CLIENT is None:
        return _v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

    image_bytes = await image.read()
    if not image_bytes:
        return _v1_error("COM_001", "이미지 파일이 비어 있습니다.", status_code=400)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return _v1_error("COM_001", "이미지 파일이 너무 큽니다 (최대 10MB).", status_code=400)
    mime_type = image.content_type or "image/jpeg"
    if mime_type not in ALLOWED_MIME_TYPES:
        return _v1_error("COM_001", f"지원하지 않는 이미지 형식: {mime_type}", status_code=400)

    try:
        analysis = await asyncio.to_thread(
            analyze_food_image_bytes,
            CLIENT,
            CONFIG.gemini_model,
            image_bytes,
            mime_type,
        )
    except Exception as e:
        return _v1_error("AI_003", f"음식 이미지 분석 실패: {e}", status_code=500)

    ingredients = []
    for item in analysis.get("추정_식재료", []):
        if not isinstance(item, dict):
            continue
        code = _map_ingredient_code(str(item.get("재료", "")).strip())
        if not code:
            continue
        ingredients.append(
            {
                "ingredientCode": code,
                "confidence": float(item.get("신뢰도", 0.5)),
            }
        )
    return _v1_success(
        {
            "requestId": requestId,
            "foodName": analysis.get("음식명"),
            "ingredients": ingredients,
            "notes": analysis.get("주의사항"),
        }
    )


app.include_router(api_v1_router)


