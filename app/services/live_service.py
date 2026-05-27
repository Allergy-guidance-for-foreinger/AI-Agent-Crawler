"""Live 라우터에서 사용하는 서비스 클래스."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests
from app.config.runtime import RuntimeContext
from app.domain.entities import FoodImageQuery, FoodTextQuery, MenuCrawlQuery, SpringForwardPayload
from app.repositories.ai_repository import AIRepository
from app.repositories.crawl_repository import CrawlRepository
from app.repositories.spring_repository import SpringRepository
from app.services.menu_analysis_builder import (
    DEFAULT_ALLERGEN_CONFIDENCE,
    build_menu_analysis_failed_result,
    build_menu_analysis_success_result,
    strip_internal_analysis_fields,
)
from app.domain.image.agent import (
    analyze_food_image_bytes,
    extract_confidence_from_image_analysis,
    extract_food_name_from_image_analysis,
    extract_food_name_reason_from_image_analysis,
)

logger = logging.getLogger(__name__)


class LiveService:
    """라우터에서 호출할 비즈니스 로직 진입점."""

    def __init__(
        self,
        ctx: RuntimeContext,
        crawl_repo: CrawlRepository | None = None,
        ai_repo: AIRepository | None = None,
        spring_repo: SpringRepository | None = None,
    ):
        self.cfg = ctx.config
        self.client = ctx.client
        self.crawl_repo = crawl_repo or CrawlRepository()
        self.ai_repo = ai_repo or AIRepository()
        self.spring_repo = spring_repo or SpringRepository()

    def run_weekly_crawl_once(self) -> dict[str, Any]:
        return self.crawl_repo.run_weekly_crawl_once(self.cfg, self.client)

    def load_menu_table_for_source(self, cafeteria_name: str, source_url: str):
        query = MenuCrawlQuery(cafeteria_name=cafeteria_name, source_url=source_url)
        return self.crawl_repo.load_menu_table_for_source(query)

    def build_daily_meals(
        self,
        *,
        cafeteria_name: str,
        table: Any,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        return self.crawl_repo.build_daily_meals(
            cafeteria_name=cafeteria_name,
            table=table,
            start=start,
            end=end,
        )

    def analyze_food_text(self, food_name: str) -> dict[str, Any]:
        query = FoodTextQuery(food_name=food_name)
        return self.ai_repo.analyze_food_text(self.client, self.cfg.gemini_model, query.food_name)

    def identify_food_from_image(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
        query = FoodImageQuery(image_bytes=image_bytes, mime_type=mime_type)
        return self.ai_repo.identify_food_from_image(
            self.client,
            self.cfg.gemini_model,
            query.image_bytes,
            query.mime_type,
        )

    def extract_menu_text_from_image(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
        query = FoodImageQuery(image_bytes=image_bytes, mime_type=mime_type)
        return self.ai_repo.extract_menu_text_from_image(
            self.client,
            self.cfg.gemini_model,
            query.image_bytes,
            query.mime_type,
        )

    def translate_text(self, source_lang: str, target_lang: str, text: str) -> str:
        return self.ai_repo.translate_text(
            self.client,
            self.cfg.gemini_model,
            source_lang,
            target_lang,
            text,
        )

    def describe_menu(self, menu_id: int, menu_name: str) -> dict[str, Any]:
        description = self.ai_repo.describe_food(
            self.client,
            self.cfg.gemini_model,
            menu_name,
        )
        return {
            "menuId": menu_id,
            "menuName": menu_name,
            "description": description,
            "modelName": "gemini",
            "modelVersion": self.cfg.gemini_model,
        }

    async def analyze_food_image_with_language(
        self,
        image_bytes: bytes,
        mime_type: str,
        language: str,
    ) -> dict[str, Any]:
        """음식 이미지를 분석하고 요청 언어에 맞는 식별 결과를 반환합니다."""
        analysis = await asyncio.to_thread(
            analyze_food_image_bytes,
            self.client,
            self.cfg.gemini_model,
            image_bytes,
            mime_type,
            None,
        )
        korean_name = extract_food_name_from_image_analysis(analysis)
        korean_reason = extract_food_name_reason_from_image_analysis(analysis)
        confidence = extract_confidence_from_image_analysis(analysis)

        async def _localize_name() -> dict[str, str | None]:
            if not korean_name:
                return {"translation": None, "pronunciation": None}
            if language == "ko":
                return {"translation": korean_name, "pronunciation": korean_name}
            return await asyncio.to_thread(
                self.ai_repo.localize_food_name,
                self.client,
                self.cfg.gemini_model,
                language,
                korean_name,
            )

        async def _maybe_translate_reason(text: str | None) -> str | None:
            if not text:
                return None
            if language == "ko":
                return text
            return await asyncio.to_thread(self.translate_text, "ko", language, text)

        localized_name, localized_reason = await asyncio.gather(
            _localize_name(),
            _maybe_translate_reason(korean_reason),
        )

        return {
            "identifiedFoodKoreanName": korean_name,
            "identifiedFoodTranslationName": localized_name.get("translation"),
            "identifiedFoodPronunciationName": localized_name.get("pronunciation"),
            "identifiedFoodNameReason": localized_reason,
            "confidence": confidence,
            "modelName": "gemini",
            "modelVersion": self.cfg.gemini_model,
        }

    def map_ingredient_code(self, token: str) -> str | None:
        return self.ai_repo.map_ingredient_code(token)

    def map_allergy_code(self, token: str) -> str | None:
        return self.ai_repo.map_allergy_code(token)

    def forward_to_spring(
        self,
        *,
        url: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        forward = SpringForwardPayload(url=url, payload=payload)
        return self.spring_repo.post_json(
            url=forward.url,
            payload=forward.payload,
            token=self.cfg.spring_api_token,
            api_key=self.cfg.spring_api_key,
        )

    async def analyze_menus(
        self,
        menus: list[Any],
        *,
        max_concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _analyze_single_menu(target) -> dict[str, Any]:
            try:
                async with semaphore:
                    analysis = await asyncio.to_thread(self.analyze_food_text, target.menuName)
                result = build_menu_analysis_success_result(
                    menu_id=target.menuId,
                    menu_name=target.menuName,
                    model_name="gemini",
                    model_version=self.cfg.gemini_model,
                    analysis=analysis,
                    allergen_confidence=DEFAULT_ALLERGEN_CONFIDENCE,
                )
                unmapped = result.get("_unmappedAllergenNames") or []
                if unmapped:
                    logger.warning(
                        "menuId=%s menuName=%s unmapped allergen labels: %s",
                        target.menuId,
                        target.menuName,
                        unmapped,
                    )
                return strip_internal_analysis_fields(result)
            except Exception as e:
                return build_menu_analysis_failed_result(
                    menu_id=target.menuId,
                    menu_name=target.menuName,
                    model_name="gemini",
                    model_version=self.cfg.gemini_model,
                )

        return await asyncio.gather(*[_analyze_single_menu(target) for target in menus])

    async def translate_menus(
        self,
        menus: list[Any],
        *,
        target_languages: list[str],
        max_concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _translate_single_menu(menu) -> dict[str, Any]:
            translations: list[dict[str, str]] = []
            translation_errors: list[dict[str, str]] = []

            async def _translate_one_language(lang: str) -> tuple[str, str | None, str | None]:
                lang_code = lang.strip()
                if not lang_code:
                    return "", None, None
                try:
                    async with semaphore:
                        translated = await asyncio.to_thread(
                            self.translate_text,
                            "ko",
                            lang_code,
                            menu.menuName,
                        )
                    return lang_code, translated, None
                except Exception as e:
                    return lang_code, None, str(e)

            results_by_lang = await asyncio.gather(
                *[_translate_one_language(lang) for lang in target_languages]
            )
            for lang_code, translated, error_reason in results_by_lang:
                if not lang_code:
                    continue
                if translated is not None:
                    translations.append({"langCode": lang_code, "translatedName": translated})
                    continue
                if error_reason is not None:
                    logger.warning("translation failed menuId=%s lang=%s: %s", menu.menuId, lang_code, error_reason)
                    translation_errors.append({"langCode": lang_code, "reason": error_reason})
            return {
                "menuId": menu.menuId,
                "sourceName": menu.menuName,
                "translations": translations,
                "translationErrors": translation_errors,
            }

        return await asyncio.gather(*[_translate_single_menu(menu) for menu in menus])
