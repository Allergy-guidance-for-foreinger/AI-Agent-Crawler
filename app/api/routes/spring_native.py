"""Spring WebClient가 직접 파싱하는 비래핑(unwrapped) 엔드포인트."""

from __future__ import annotations

import asyncio
import logging

import requests
from fastapi import APIRouter, Body, Request
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from app.config.runtime import API_V1_PREFIX, RuntimeContext
from app.schemas.api_models import (
    PythonMealCrawlRequest,
    PythonMenuAnalysisRequest,
    PythonMenuDescribeListRequest,
    PythonMenuDescribeListResponse,
    PythonMenuDescribeRequest,
    TextListTranslationRequest,
    TextListTranslationResponse,
)
from app.services.live_service import LiveService
from app.common.service_ops import (
    CrawlSourceUpstreamError,
    sanitize_url_for_log,
    v1_error,
    validate_accept_language,
)

logger = logging.getLogger(__name__)


class FreeTranslationRequest(BaseModel):
    sourceLang: str = Field(..., min_length=1)
    targetLang: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


def create_spring_native_router(ctx: RuntimeContext) -> APIRouter:
    service = LiveService(ctx)
    cfg = service.cfg
    client = service.client
    router = APIRouter(prefix=API_V1_PREFIX)

    @router.post(
        "/crawl/meals",
        tags=["spring-native"],
        summary="식단 크롤링 (unwrapped)",
        operation_id="springNativeCrawlMeals",
    )
    def crawl_meals_native(
        request: Request,
        payload: PythonMealCrawlRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)

        try:
            meals = service.crawl_daily_meals(
                cafeteria_name=payload.cafeteriaName,
                source_url=payload.sourceUrl,
                start=payload.startDate,
                end=payload.endDate,
            )
        except RuntimeError as e:
            logger.warning("crawl bad request cafeteria=%s reason=%s", payload.cafeteriaName, e)
            return v1_error("PYM_400", "요청 식단 조회 조건이 유효하지 않거나 데이터가 없습니다.", status_code=400)
        except (CrawlSourceUpstreamError, requests.exceptions.RequestException, OSError) as e:
            logger.warning(
                "upstream crawl source unavailable source=%s cafeteriaName=%s: %s",
                sanitize_url_for_log(payload.sourceUrl),
                payload.cafeteriaName,
                e,
            )
            return v1_error("PYM_502", "외부 크롤링 소스 조회에 실패했습니다. 잠시 후 다시 시도해주세요.", status_code=502)

        return {
            "schoolName": payload.schoolName,
            "cafeteriaName": payload.cafeteriaName,
            "sourceUrl": payload.sourceUrl,
            "startDate": payload.startDate.isoformat(),
            "endDate": payload.endDate.isoformat(),
            "meals": meals,
        }

    @router.post(
        "/menus/analyze",
        tags=["spring-native"],
        summary="메뉴 AI 분석 (unwrapped)",
        operation_id="springNativeAnalyzeMenus",
    )
    async def analyze_menus_native(
        request: Request,
        payload: PythonMenuAnalysisRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)
        if client is None:
            return v1_error("AI_001", "AI 서비스가 구성되지 않았습니다.", status_code=500)

        results = await service.analyze_menus(payload.menus, max_concurrency=cfg.ai_max_concurrent_tasks)
        return {"results": results}

    @router.post(
        "/menus/describe",
        tags=["실사용 API"],
        summary="메뉴명 한국어 설명 (unwrapped)",
        operation_id="springNativeDescribeMenu",
    )
    async def describe_menu_native(
        request: Request,
        payload: PythonMenuDescribeRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)
        if client is None:
            return v1_error("AI_001", "AI 서비스가 구성되지 않았습니다.", status_code=500)
        try:
            result = await asyncio.to_thread(
                service.describe_menu,
                payload.menuId,
                payload.menuName.strip(),
            )
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
            logger.warning("describe menus list runtime error: %s", e)
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)
        except (genai_errors.ClientError, genai_errors.ServerError):
            logger.warning("upstream gemini describe failed")
            return v1_error(
                "PYM_502",
                "외부 AI 서비스 호출에 실패했습니다. 잠시 후 다시 시도해주세요.",
                status_code=502,
            )
        except Exception:
            logger.exception("unexpected describe menu error")
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)
        return result

    @router.post(
        "/menus/describe/list",
        tags=["실사용 API"],
        summary="메뉴 목록 설명 (unwrapped)",
        description="langCode와 메뉴 목록을 받아 각 메뉴 설명 결과(results)를 반환합니다.",
        operation_id="springNativeDescribeMenusList",
        response_model=PythonMenuDescribeListResponse,
    )
    async def describe_menus_list_native(
        request: Request,
        payload: PythonMenuDescribeListRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)
        if client is None:
            return v1_error("AI_001", "AI 서비스가 구성되지 않았습니다.", status_code=500)
        try:
            results = await service.describe_menus(
                payload.menus,
                lang_code=payload.langCode.strip(),
                max_concurrency=cfg.ai_max_concurrent_tasks,
            )
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
            logger.warning("describe menus list runtime error: %s", e)
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)
        except (genai_errors.ClientError, genai_errors.ServerError):
            logger.warning("upstream gemini describe list failed")
            return v1_error(
                "PYM_502",
                "외부 AI 서비스 호출에 실패했습니다. 잠시 후 다시 시도해주세요.",
                status_code=502,
            )
        except Exception:
            logger.exception("unexpected describe menus list error")
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)
        return {"results": results}

    @router.post(
        "/translations",
        tags=["실사용 API"],
        summary="자유 텍스트 번역",
        operation_id="springNativeTranslateText",
    )
    async def translate_text_native(
        request: Request,
        payload: FreeTranslationRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)
        if client is None:
            return v1_error("AI_001", "AI 서비스가 구성되지 않았습니다.", status_code=500)

        translated = await asyncio.to_thread(
            service.translate_text,
            payload.sourceLang,
            payload.targetLang,
            payload.text,
        )
        return {"translatedText": translated}

    @router.post(
        "/translations/list",
        tags=["실사용 API"],
        summary="문자열 목록 일괄 번역 (unwrapped)",
        description="재료 목록(ingredientCode, text)을 번역해 results 배열로 반환합니다.",
        operation_id="springNativeTranslateTextList",
        response_model=TextListTranslationResponse,
    )
    async def translate_text_list_native(
        request: Request,
        payload: TextListTranslationRequest = Body(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return v1_error("COM_001", str(e), status_code=400)
        if client is None:
            return v1_error("AI_001", "AI 서비스가 구성되지 않았습니다.", status_code=500)
        try:
            return await asyncio.to_thread(
                _translate_and_pack_results,
                service,
                payload,
            )
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
            return v1_error("PYM_500", str(e), status_code=500)
        except (genai_errors.ClientError, genai_errors.ServerError):
            logger.warning("upstream gemini translate list failed")
            return v1_error(
                "PYM_502",
                "외부 AI 서비스 호출에 실패했습니다. 잠시 후 다시 시도해주세요.",
                status_code=502,
            )
        except Exception:
            logger.exception("unexpected translate text list error")
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)

    return router


def _translate_and_pack_results(service: LiveService, payload: TextListTranslationRequest) -> dict[str, list[dict[str, str]]]:
    translated = service.translate_text_list(
        payload.sourceLang.strip(),
        payload.targetLang.strip(),
        [item.text.strip() for item in payload.ingredients],
    )
    return {
        "results": [
            {
                "ingredientCode": payload.ingredients[idx].ingredientCode.strip(),
                "translatedText": translated[idx],
            }
            for idx in range(len(payload.ingredients))
        ]
    }
