"""`/api/v1` 요청 DTO 모음."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


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
