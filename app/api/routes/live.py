"""FastAPI `/api/v1` 라우터 (식단/분석/번역 3개 API만 제공)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests
from fastapi import APIRouter, Body, File, Form, Request, UploadFile
from google.genai import errors as genai_errors

from app.config.runtime import ALLOWED_MIME_TYPES, API_V1_PREFIX, MAX_IMAGE_SIZE, RuntimeContext
from app.schemas.api_models import (
    ApiErrorResponse,
    ApiSuccessResponse,
    PythonMealCrawlResponse,
    PythonMealCrawlRequest,
    PythonMenuAnalysisResponse,
    PythonMenuImageAnalysisResponse,
    PythonMenuAnalysisRequest,
    PythonMenuAnalysisTargetDto,
    PythonMenuDescribeRequest,
    PythonMenuDescribeResponse,
    PythonMenuOcrResponse,
    PythonMenuTranslationResponse,
    PythonMenuTranslationRequest,
)
from app.schemas.openapi_examples import (
    AI_KEY_MISSING_EXAMPLE,
    MEAL_CRAWL_ERROR_BAD_CONDITION_EXAMPLE,
    MEAL_CRAWL_ERROR_UPSTREAM_EXAMPLE,
    MEAL_CRAWL_REQUEST_OPENAPI_EXAMPLES,
    MEAL_CRAWL_SUCCESS_EXAMPLE,
    FOOD_DESCRIBE_REQUEST_OPENAPI_EXAMPLES,
    FOOD_DESCRIBE_SUCCESS_EXAMPLE,
    MENU_ANALYZE_REQUEST_OPENAPI_EXAMPLES,
    MENU_ANALYZE_SUCCESS_EXAMPLE,
    MENU_TRANSLATE_REQUEST_OPENAPI_EXAMPLES,
    MENU_TRANSLATE_SUCCESS_EXAMPLE,
    VALIDATION_ERROR_EXAMPLE,
    V1_INTERNAL_SERVER_ERROR_EXAMPLE,
)
from app.services.live_service import LiveService
from app.common.service_ops import (
    CrawlSourceUpstreamError,
    sanitize_url_for_log,
    v1_error,
    v1_success,
    validate_accept_language,
    normalize_request_language,
)

logger = logging.getLogger(__name__)


def _v1_bad_request(msg: str):
    return v1_error("COM_001", msg, status_code=400)


def _validate_image_upload_v1(image_bytes: bytes, mime_type: str) -> tuple[bool, Any]:
    if not image_bytes:
        return False, _v1_bad_request("이미지 파일이 비어 있습니다.")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return False, v1_error("COM_001", "이미지 파일이 너무 큽니다 (최대 10MB).", status_code=413)
    if mime_type not in ALLOWED_MIME_TYPES:
        return False, _v1_bad_request(f"지원하지 않는 이미지 형식: {mime_type}")
    return True, None


def _safe_float(value: object, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _map_ocr_exception_to_v1_error(exc: Exception):
    if isinstance(exc, RuntimeError) and "GEMINI_API_KEY is not set" in str(exc):
        return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
    if isinstance(exc, (genai_errors.ClientError, genai_errors.ServerError)):
        logger.warning("upstream gemini call failed: %s", exc)
        return v1_error("PYM_502", "외부 AI 서비스 호출에 실패했습니다. 잠시 후 다시 시도해주세요.", status_code=502)
    logger.exception("unexpected OCR/analyze-from-ocr error")
    return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)


def create_v1_router(ctx: RuntimeContext) -> APIRouter:
    service = LiveService(ctx)
    cfg = service.cfg
    client = service.client
    router = APIRouter(prefix=API_V1_PREFIX)

    @router.post(
        "/python/meals/crawl",
        tags=["실사용 API"],
        summary="식단 기간 조회/크롤링",
        description="학교/식당/기간 조건으로 식단을 조회하고 메뉴 목록을 표준 DTO로 반환합니다.",
        operation_id="crawlMealsV1",
        response_model=ApiSuccessResponse[PythonMealCrawlResponse],
        responses={
            200: {
                "description": "조회 성공",
                "content": {
                    "application/json": {
                        "examples": {
                            "식단포함": {
                                "summary": "meals에 일자별 메뉴 포함",
                                "value": MEAL_CRAWL_SUCCESS_EXAMPLE,
                            }
                        }
                    }
                },
            },
            400: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "검증실패": {"summary": "스키마/헤더 검증 오류", "value": VALIDATION_ERROR_EXAMPLE},
                            "조건불가": {
                                "summary": "식당/URL 조건 오류",
                                "value": MEAL_CRAWL_ERROR_BAD_CONDITION_EXAMPLE,
                            },
                        }
                    }
                },
            },
            502: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "업스트림실패": {"summary": "외부 크롤 소스 실패", "value": MEAL_CRAWL_ERROR_UPSTREAM_EXAMPLE},
                        }
                    }
                },
            },
            500: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "서버오류": {
                                "summary": "처리 중 내부 오류(문서 예시)",
                                "value": V1_INTERNAL_SERVER_ERROR_EXAMPLE,
                            },
                        }
                    }
                },
            },
        },
    )
    def crawl_meals_v1(
        request: Request,
        payload: PythonMealCrawlRequest = Body(..., openapi_examples=MEAL_CRAWL_REQUEST_OPENAPI_EXAMPLES),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))
        if payload.startDate > payload.endDate:
            return _v1_bad_request("startDate는 endDate보다 이후일 수 없습니다.")
        try:
            table = service.load_menu_table_for_source(payload.cafeteriaName, payload.sourceUrl)
        except RuntimeError as e:
            logger.warning("crawl bad request cafeteria=%s reason=%s", payload.cafeteriaName, e)
            return v1_error("PYM_400", "요청 식단 조회 조건이 유효하지 않거나 데이터가 없습니다.", status_code=400)
        except (CrawlSourceUpstreamError, requests.exceptions.RequestException, OSError) as e:
            logger.warning("upstream crawl source unavailable source=%s cafeteriaName=%s: %s", sanitize_url_for_log(payload.sourceUrl), payload.cafeteriaName, e)
            return v1_error("PYM_502", "외부 크롤링 소스 조회에 실패했습니다. 잠시 후 다시 시도해주세요.", status_code=502)
        meals = service.build_daily_meals(cafeteria_name=payload.cafeteriaName, table=table, start=payload.startDate, end=payload.endDate)
        return v1_success({"schoolName": payload.schoolName, "cafeteriaName": payload.cafeteriaName, "sourceUrl": payload.sourceUrl, "startDate": payload.startDate.isoformat(), "endDate": payload.endDate.isoformat(), "meals": meals})

    @router.post(
        "/python/menus/analyze",
        tags=["실사용 API"],
        summary="메뉴 텍스트 AI 분석",
        description="메뉴명 리스트를 받아 재료 코드/신뢰도 추정 결과를 반환합니다.",
        operation_id="analyzeMenusV1",
        response_model=ApiSuccessResponse[PythonMenuAnalysisResponse],
        responses={
            200: {
                "description": "분석 성공",
                "content": {
                    "application/json": {
                        "examples": {
                            "재료추정": {"summary": "ingredients 포함", "value": MENU_ANALYZE_SUCCESS_EXAMPLE},
                        }
                    }
                },
            },
            400: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "검증실패": {"value": VALIDATION_ERROR_EXAMPLE},
                        }
                    }
                },
            },
            500: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "키미설정": {"summary": "GEMINI 미설정", "value": AI_KEY_MISSING_EXAMPLE},
                        }
                    }
                },
            },
        },
    )
    async def analyze_menus_v1(
        request: Request,
        payload: PythonMenuAnalysisRequest = Body(..., openapi_examples=MENU_ANALYZE_REQUEST_OPENAPI_EXAMPLES),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
        results = await service.analyze_menus(payload.menus, max_concurrency=cfg.ai_max_concurrent_tasks)
        return v1_success({"results": results})

    @router.post(
        "/python/menus/describe",
        tags=["실사용 API"],
        summary="메뉴명 한국어 설명",
        description="menuId·menuName을 받아 해당 음식을 한국어로 설명합니다.",
        operation_id="describeMenuV1",
        response_model=ApiSuccessResponse[PythonMenuDescribeResponse],
        responses={
            200: {
                "description": "설명 생성 성공",
                "content": {
                    "application/json": {
                        "examples": {
                            "기본": {"value": FOOD_DESCRIBE_SUCCESS_EXAMPLE},
                        }
                    }
                },
            },
            400: {"model": ApiErrorResponse},
            500: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {"키미설정": {"value": AI_KEY_MISSING_EXAMPLE}},
                    }
                },
            },
        },
    )
    async def describe_food_v1(
        request: Request,
        payload: PythonMenuDescribeRequest = Body(
            ..., openapi_examples=FOOD_DESCRIBE_REQUEST_OPENAPI_EXAMPLES
        ),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
        try:
            result = await asyncio.to_thread(
                service.describe_menu,
                payload.menuId,
                payload.menuName.strip(),
            )
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
            return v1_error("PYM_500", str(e), status_code=500)
        except (genai_errors.ClientError, genai_errors.ServerError):
            logger.warning("upstream gemini describe failed")
            return v1_error(
                "PYM_502",
                "외부 AI 서비스 호출에 실패했습니다. 잠시 후 다시 시도해주세요.",
                status_code=502,
            )
        except Exception:
            logger.exception("unexpected describe food error")
            return v1_error("PYM_500", "요청 처리 중 내부 오류가 발생했습니다.", status_code=500)
        return v1_success(result)

    @router.post(
        "/python/menus/ocr",
        tags=["v1"],
        summary="메뉴판 OCR 추출",
        description="메뉴판 이미지를 OCR 관점으로 읽어 메뉴명 목록을 추출합니다.",
        operation_id="ocrMenuBoardV1",
        response_model=ApiSuccessResponse[PythonMenuOcrResponse],
        responses={
            400: {"model": ApiErrorResponse},
            500: {"model": ApiErrorResponse},
            413: {"model": ApiErrorResponse},
        },
    )
    async def ocr_menu_board_v1(
        request: Request,
        image: UploadFile = File(...),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))

        image_bytes = await image.read()
        mime_type = image.content_type or "image/jpeg"
        valid, err = _validate_image_upload_v1(image_bytes, mime_type)
        if not valid:
            return err
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

        try:
            parsed = await asyncio.to_thread(service.extract_menu_text_from_image, image_bytes, mime_type)
        except Exception as exc:
            return _map_ocr_exception_to_v1_error(exc)
        return v1_success(
            {
                "rawText": parsed.get("rawText", ""),
                "menus": [{"menuName": name} for name in (parsed.get("menuNames") or [])],
            }
        )

    @router.post(
        "/python/menus/analyze-from-ocr",
        tags=["v1"],
        summary="메뉴판 OCR 후 메뉴 분석",
        description="메뉴판 OCR 결과를 기반으로 메뉴 분석을 연속 수행합니다.",
        operation_id="analyzeMenusFromOcrV1",
        response_model=ApiSuccessResponse[PythonMenuAnalysisResponse],
        responses={
            400: {"model": ApiErrorResponse},
            500: {"model": ApiErrorResponse},
            413: {"model": ApiErrorResponse},
        },
    )
    async def analyze_menus_from_ocr_v1(
        request: Request,
        image: UploadFile = File(...),
        startMenuId: int = Form(default=1, ge=1),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))

        image_bytes = await image.read()
        mime_type = image.content_type or "image/jpeg"
        valid, err = _validate_image_upload_v1(image_bytes, mime_type)
        if not valid:
            return err
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

        try:
            parsed = await asyncio.to_thread(service.extract_menu_text_from_image, image_bytes, mime_type)
        except Exception as exc:
            return _map_ocr_exception_to_v1_error(exc)
        menu_names = parsed.get("menuNames") or []
        targets = [
            PythonMenuAnalysisTargetDto(menuId=startMenuId + idx, menuName=menu_name)
            for idx, menu_name in enumerate(menu_names)
        ]
        if not targets:
            return v1_success({"results": []})
        results = await service.analyze_menus(targets, max_concurrency=cfg.ai_max_concurrent_tasks)
        return v1_success({"results": results})

    @router.post(
        "/python/menus/analyze-image",
        tags=["실사용 API"],
        summary="이미지 기반 메뉴 AI 분석",
        description=(
            "음식 이미지에서 음식명·판단 근거·모델 신뢰도(confidence)를 반환합니다. "
            "요청은 language와 image를 받으며, 응답은 요청 language 기준으로 내려갑니다."
        ),
        operation_id="analyzeMenuImageV1",
        response_model=ApiSuccessResponse[PythonMenuImageAnalysisResponse],
        responses={
            400: {"model": ApiErrorResponse},
            500: {"model": ApiErrorResponse},
            413: {"model": ApiErrorResponse},
        },
    )
    async def analyze_menu_image_v1(
        request: Request,
        image: UploadFile = File(...),
        language: str = Form(default="ko"),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
            lang_code = normalize_request_language(language)
        except ValueError as e:
            return _v1_bad_request(str(e))

        image_bytes = await image.read()
        mime_type = image.content_type or "image/jpeg"
        valid, err = _validate_image_upload_v1(image_bytes, mime_type)
        if not valid:
            return err
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)

        try:
            result = await service.analyze_food_image_with_language(
                image_bytes,
                mime_type,
                lang_code,
            )
        except Exception:
            logger.exception("analyze_menu_image_v1 failed")
            return v1_error("PYM_500", "이미지 분석 중 내부 오류가 발생했습니다.", status_code=500)
        return v1_success({"results": [result]})

    @router.post(
        "/python/menus/translate",
        tags=["v1"],
        summary="메뉴 다국어 번역",
        description="메뉴명 리스트를 요청 언어 목록으로 번역해 반환합니다.",
        operation_id="translateMenusV1",
        response_model=ApiSuccessResponse[PythonMenuTranslationResponse],
        responses={
            200: {
                "description": "번역 성공",
                "content": {
                    "application/json": {
                        "examples": {
                            "다국어": {"summary": "translations 배열", "value": MENU_TRANSLATE_SUCCESS_EXAMPLE},
                        }
                    }
                },
            },
            400: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "검증실패": {"value": VALIDATION_ERROR_EXAMPLE},
                        }
                    }
                },
            },
            500: {
                "model": ApiErrorResponse,
                "content": {
                    "application/json": {
                        "examples": {
                            "키미설정": {"value": AI_KEY_MISSING_EXAMPLE},
                        }
                    }
                },
            },
        },
    )
    async def translate_menus_v1(
        request: Request,
        payload: PythonMenuTranslationRequest = Body(..., openapi_examples=MENU_TRANSLATE_REQUEST_OPENAPI_EXAMPLES),
    ):
        try:
            validate_accept_language(request.headers.get("Accept-Language"))
        except ValueError as e:
            return _v1_bad_request(str(e))
        if client is None:
            return v1_error("AI_001", "GEMINI_API_KEY is not set", status_code=500)
        results = await service.translate_menus(
            payload.menus,
            target_languages=payload.targetLanguages,
            max_concurrency=cfg.ai_max_concurrent_tasks,
        )
        return v1_success({"results": results})

    return router
