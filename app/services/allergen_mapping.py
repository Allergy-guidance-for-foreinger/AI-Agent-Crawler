"""식약처 알레르기 표시 기준 토큰 → API 코드 매핑 (부분 문자열 오매칭 방지)."""

from __future__ import annotations

import re
from typing import Any

from app.config.runtime import CANONICAL_TO_INGREDIENT_CODE
from user_features.allergen_catalog import ALIAS_TO_CANONICAL, list_canonical_choices

MFDS_ALLERGEN_LABELS: tuple[str, ...] = tuple(list_canonical_choices())

ALLERGY_KEYWORD_TO_API_CODE: dict[str, str] = {
    "mackerel": "MACKEREL",
    "고등어": "MACKEREL",
    "crab": "CRAB",
    "게": "CRAB",
    "shrimp": "SHRIMP",
    "새우": "SHRIMP",
    "squid": "SQUID",
    "오징어": "SQUID",
    "shellfish": "SHELLFISH",
    "조개류": "SHELLFISH",
    "clam": "CLAM",
    "조개": "CLAM",
    "mussel": "MUSSEL",
    "홍합": "MUSSEL",
    "oyster": "OYSTER",
    "굴": "OYSTER",
    "lobster": "LOBSTER",
    "랍스터": "LOBSTER",
    "scallop": "SCALLOP",
    "가리비": "SCALLOP",
    "pork": "PORK",
    "돼지고기": "PORK",
    "돼지": "PORK",
    "제육": "PORK",
    "chicken": "CHICKEN",
    "닭고기": "CHICKEN",
    "닭": "CHICKEN",
    "치킨": "CHICKEN",
    "beef": "BEEF",
    "쇠고기": "BEEF",
    "소고기": "BEEF",
    "egg": "EGG",
    "난류": "EGG",
    "계란": "EGG",
    "달걀": "EGG",
    "milk": "MILK",
    "dairy": "MILK",
    "우유": "MILK",
    "유제품": "MILK",
    "peanut": "PEANUT",
    "땅콩": "PEANUT",
    "soybean": "SOYBEAN",
    "soy": "SOYBEAN",
    "대두": "SOYBEAN",
    "wheat": "WHEAT",
    "밀": "WHEAT",
    "buckwheat": "BUCKWHEAT",
    "메밀": "BUCKWHEAT",
    "oats": "OATS",
    "귀리": "OATS",
    "rye": "RYE",
    "호밀": "RYE",
    "barley": "BARLEY",
    "보리": "BARLEY",
    "tree nut": "TREE_NUT",
    "tree nuts": "TREE_NUT",
    "견과류": "TREE_NUT",
    "walnut": "WALNUT",
    "호두": "WALNUT",
    "almond": "ALMOND",
    "아몬드": "ALMOND",
    "hazelnut": "HAZELNUT",
    "헤이즐넛": "HAZELNUT",
    "cashew": "CASHEW",
    "캐슈너트": "CASHEW",
    "pistachio": "PISTACHIO",
    "피스타치오": "PISTACHIO",
    "pecan": "PECAN",
    "피칸": "PECAN",
    "brazil nut": "BRAZIL_NUT",
    "브라질너트": "BRAZIL_NUT",
    "macadamia": "MACADAMIA",
    "마카다미아": "MACADAMIA",
    "pine nut": "PINE_NUT",
    "잣": "PINE_NUT",
    "peach": "PEACH",
    "복숭아": "PEACH",
    "mango": "MANGO",
    "망고": "MANGO",
    "avocado": "AVOCADO",
    "아보카도": "AVOCADO",
    "banana": "BANANA",
    "바나나": "BANANA",
    "kiwi": "KIWI",
    "키위": "KIWI",
    "tomato": "TOMATO",
    "토마토": "TOMATO",
    "celery": "CELERY",
    "셀러리": "CELERY",
    "mustard": "MUSTARD",
    "머스타드": "MUSTARD",
    "sulfites": "SULFITES",
    "아황산류": "SULFITES",
    "sesame": "SESAME",
    "참깨": "SESAME",
    "lupin": "LUPIN",
    "루핀": "LUPIN",
    "latex": "LATEX_RELATED",
    "라텍스": "LATEX_RELATED",
}

ALLERGY_API_CODES: frozenset[str] = frozenset(ALLERGY_KEYWORD_TO_API_CODE.values())


def _code_from_canonical_label(label: str) -> str | None:
    code = CANONICAL_TO_INGREDIENT_CODE.get(label)
    if code and code in ALLERGY_API_CODES:
        return code
    return None


def map_allergy_code(token: str) -> str | None:
    """알레르기 표준 토큰 → API allergyCode. 별칭·정확 일치만 허용."""
    normalized = token.strip()
    if not normalized:
        return None

    direct = _code_from_canonical_label(normalized)
    if direct:
        return direct

    normalized_upper = normalized.upper().replace("-", "_").replace(" ", "_")
    if normalized_upper in ALLERGY_API_CODES:
        return normalized_upper

    alias_key = normalized.lower() if normalized.isascii() else normalized
    canonical = ALIAS_TO_CANONICAL.get(normalized) or ALIAS_TO_CANONICAL.get(alias_key)
    if canonical:
        return _code_from_canonical_label(canonical)

    if normalized in ALLERGY_KEYWORD_TO_API_CODE:
        code = ALLERGY_KEYWORD_TO_API_CODE[normalized]
        return code if code in ALLERGY_API_CODES else None
    lowered = normalized.lower()
    if lowered in ALLERGY_KEYWORD_TO_API_CODE:
        code = ALLERGY_KEYWORD_TO_API_CODE[lowered]
        return code if code in ALLERGY_API_CODES else None

    for keyword, code in sorted(
        ALLERGY_KEYWORD_TO_API_CODE.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not keyword.isascii() or len(keyword) < 2:
            continue
        if code not in ALLERGY_API_CODES:
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return code
    return None


def map_ingredient_code(token: str) -> str | None:
    """재료명 → (선택) 알레르기 API 코드. 정확 일치·별칭만."""
    return map_allergy_code(token)


def format_mfds_labels_for_prompt() -> str:
    return ", ".join(MFDS_ALLERGEN_LABELS)


def build_ingredient_results(
    ingredients_ko: list[Any],
    *,
    base_confidence: float = 0.95,
    confidence_decay: float = 0.07,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """모델 재료 목록 → API ingredients (이름 전부 + 선택적 코드)."""
    results: list[dict[str, Any]] = []
    for idx, raw in enumerate(ingredients_ko or []):
        name = str(raw).strip()
        if not name:
            continue
        code = map_ingredient_code(name)
        results.append(
            {
                "ingredientName": name,
                "ingredientCode": code,
                "confidence": round(
                    max(min_confidence, base_confidence - (idx * confidence_decay)),
                    2,
                ),
            }
        )
    return results


def build_allergy_results(
    allergens_ko: list[Any],
    *,
    fallback_confidence: float = 0.8,
) -> tuple[list[dict[str, Any]], list[str]]:
    """모델 알레르기 목록 → API allergies + 매핑 실패한 표준명."""
    allergies: list[dict[str, Any]] = []
    unmapped: list[str] = []
    dedup: set[str] = set()
    for allergen in allergens_ko or []:
        if not isinstance(allergen, dict):
            continue
        label = str(allergen.get("name", "")).strip()
        if not label:
            continue
        code = map_allergy_code(label)
        if not code or code in dedup:
            if label and not code:
                unmapped.append(label)
            continue
        dedup.add(code)
        allergies.append({"allergyCode": code, "confidence": fallback_confidence})
    return allergies, unmapped
