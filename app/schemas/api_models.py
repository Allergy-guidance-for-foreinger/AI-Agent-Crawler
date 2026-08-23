"""`/api/v1/python/*` DTO 모음."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import Generic, Optional, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from app.domain.crawler.knu_menu import (
    MAX_WEEK_FETCHES,
    SHOP_NAMES,
    is_knu_host,
    week_sel_dates_covering,
)
from app.domain.crawler.kumoh_menu import normalize_kumoh_cafeteria_name

T = TypeVar("T")


class PythonMealCrawlRequest(BaseModel):
    schoolName: str = Field(..., min_length=1)
    cafeteriaName: str = Field(
        ...,
        min_length=1,
        description=(
            "식당명. 금오: 일품식당/정찬식당/분식당(구명칭 학생식당·교직원식당 자동 치환). "
            "경북대: 정보센터식당, 복지관 교직원식당, 카페테리아 첨성, GP감꽃식당, "
            "공학관교직원식당(외부업체)."
        ),
    )
    sourceUrl: str = Field(..., min_length=1)
    startDate: date
    endDate: date

    @model_validator(mode="after")
    def validate_date_range_and_cafeteria(self):
        if self.startDate > self.endDate:
            raise ValueError("startDate는 endDate보다 이후일 수 없습니다.")
        if len(week_sel_dates_covering(self.startDate, self.endDate)) > MAX_WEEK_FETCHES:
            raise ValueError(
                f"조회 기간은 최대 {MAX_WEEK_FETCHES}주까지 허용됩니다."
            )
        host = urlparse(self.sourceUrl).hostname
        if not is_knu_host(host):
            self.cafeteriaName = normalize_kumoh_cafeteria_name(self.cafeteriaName)
        else:
            name = self.cafeteriaName.strip()
            if name not in SHOP_NAMES.values():
                raise ValueError(
                    "지원하지 않는 경북대 식당명입니다. "
                    "정보센터식당, 복지관 교직원식당, 카페테리아 첨성, GP감꽃식당, "
                    "공학관교직원식당(외부업체) 중 하나를 사용하세요."
                )
            self.cafeteriaName = name
        return self


class PythonMenuAnalysisTargetDto(BaseModel):
    menuId: int
    menuName: str = Field(..., min_length=1)


class PythonMenuAnalysisRequest(BaseModel):
    menus: list[PythonMenuAnalysisTargetDto] = Field(..., min_length=1)


class PythonMenuOcrMenuDto(BaseModel):
    menuName: str


class PythonMenuOcrResponse(BaseModel):
    rawText: str
    menus: list[PythonMenuOcrMenuDto]


class PythonMenuTranslationTargetDto(BaseModel):
    menuId: int
    menuName: str = Field(..., min_length=1)


class PythonMenuTranslationRequest(BaseModel):
    menus: list[PythonMenuTranslationTargetDto] = Field(..., min_length=1)
    targetLanguages: list[str] = Field(..., min_length=1)


class ApiErrorResponse(BaseModel):
    success: bool = Field(default=False, examples=[False])
    code: str = Field(..., examples=["COM_002"])
    msg: str = Field(..., examples=["요청 데이터 변환 과정에서 오류가 발생했습니다."])


class ApiSuccessResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True, examples=[True])
    data: T


class PythonCrawledMenuDto(BaseModel):
    cornerName: str
    displayOrder: int
    menuName: str


class PythonDailyMealDto(BaseModel):
    mealDate: date
    mealType: str
    menus: list[PythonCrawledMenuDto]


class PythonMealCrawlResponse(BaseModel):
    schoolName: str
    cafeteriaName: str
    sourceUrl: str
    startDate: date
    endDate: date
    meals: list[PythonDailyMealDto]


class PythonMenuIngredientResultDto(BaseModel):
    ingredientName: str = Field(..., min_length=1, description="추정 주재료명(한국어)")
    ingredientCode: Optional[str] = Field(
        default=None,
        description="식약처 알레르기 API 코드와 정확 매칭될 때만 채움",
    )
    confidence: float


class PythonMenuAllergyResultDto(BaseModel):
    allergyCode: str
    confidence: float


class PythonMenuAnalysisResultDto(BaseModel):
    """Spring PythonMenuAnalysisResultDto와 동일 필드 집합."""

    menuId: int
    menuName: str
    status: str  # SUCCESS | FAILED
    spicyLevel: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="매운맛 0(없음)~5. 실패·미추정 시 null",
    )
    modelName: str
    modelVersion: str
    ingredients: list[PythonMenuIngredientResultDto]
    allergies: list[PythonMenuAllergyResultDto] = Field(default_factory=list)


class PythonMenuAnalysisResponse(BaseModel):
    results: list[PythonMenuAnalysisResultDto]


class PythonMenuDescribeRequest(BaseModel):
    menuId: int
    menuName: str = Field(..., min_length=1, description="설명할 메뉴명")


class PythonMenuDescribeResponse(BaseModel):
    menuId: int
    menuName: str
    description: str = Field(..., description="음식에 대한 한국어 설명")
    modelName: str
    modelVersion: str


class PythonMenuDescribeListRequest(BaseModel):
    langCode: str = Field(..., min_length=1, description="응답 설명 언어 코드 (예: ko, en)")
    menus: list[PythonMenuDescribeRequest] = Field(..., min_length=1, description="설명할 메뉴 목록")

    @model_validator(mode="after")
    def validate_fields(self):
        if not self.langCode.strip():
            raise ValueError("langCode는 공백일 수 없습니다.")
        return self


class PythonMenuDescribeListResultDto(BaseModel):
    menuId: int
    description: str = Field(..., description="요청 언어 기준 메뉴 설명")


class PythonMenuDescribeListResponse(BaseModel):
    results: list[PythonMenuDescribeListResultDto]


class PythonMenuImageAnalysisResultDto(BaseModel):
    """`POST /python/menus/analyze-image` 전용 응답 항목."""

    identifiedFoodKoreanName: Optional[str] = Field(
        default=None,
        description="이미지에서 추정한 음식명(한국어)",
    )
    identifiedFoodTranslationName: Optional[str] = Field(
        default=None,
        description="요청 language로 번역한 음식명(의미)",
    )
    identifiedFoodPronunciationName: Optional[str] = Field(
        default=None,
        description="요청 language 기준 한국어 음식명 발음 표기",
    )
    identifiedFoodNameReason: Optional[str] = Field(
        default=None,
        description="해당 음식으로 판단한 시각적·맥락적 근거(요청 language)",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="추정 음식명에 대한 모델 신뢰도 0.0~1.0",
    )
    modelName: str
    modelVersion: str


class PythonMenuImageAnalysisResponse(BaseModel):
    results: list[PythonMenuImageAnalysisResultDto]


class PythonTranslatedMenuNameDto(BaseModel):
    langCode: str
    translatedName: str


class PythonMenuTranslationResultDto(BaseModel):
    menuId: int
    sourceName: str
    translations: list[PythonTranslatedMenuNameDto]


class PythonMenuTranslationResponse(BaseModel):
    results: list[PythonMenuTranslationResultDto]


class IngredientTranslationInputDto(BaseModel):
    ingredientCode: str = Field(..., min_length=1, description="원문 재료 식별 코드")
    text: str = Field(..., min_length=1, description="번역할 원문 텍스트")


class IngredientTranslationResultDto(BaseModel):
    ingredientCode: str = Field(..., min_length=1, description="요청과 동일한 재료 식별 코드")
    translatedText: str = Field(..., min_length=1, description="번역 결과 텍스트")


class TextListTranslationRequest(BaseModel):
    """재료명 목록 일괄 번역 요청."""

    sourceLang: str = Field(..., min_length=1, description="원문 언어 코드 (예: ko)")
    targetLang: str = Field(..., min_length=1, description="번역 언어 코드 (예: en)")
    ingredients: list[IngredientTranslationInputDto] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="번역할 재료 목록 (최대 100개)",
    )

    @model_validator(mode="after")
    def validate_non_empty_ingredients(self):
        if not self.sourceLang.strip() or not self.targetLang.strip():
            raise ValueError("sourceLang/targetLang은 공백일 수 없습니다.")
        for item in self.ingredients:
            if not item.ingredientCode.strip() or not item.text.strip():
                raise ValueError("ingredients의 ingredientCode/text는 비어 있을 수 없습니다.")
        return self


class TextListTranslationResponse(BaseModel):
    results: list[IngredientTranslationResultDto]
