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

# 영문 키워드 단어 경계 매칭용 (호출마다 정규식 컴파일하지 않음)
_ASCII_KEYWORD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(keyword)}\b"), code)
    for keyword, code in sorted(
        ALLERGY_KEYWORD_TO_API_CODE.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if keyword.isascii() and len(keyword) >= 2
)


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
        return ALLERGY_KEYWORD_TO_API_CODE[normalized]
    lowered = normalized.lower()
    if lowered in ALLERGY_KEYWORD_TO_API_CODE:
        return ALLERGY_KEYWORD_TO_API_CODE[lowered]

    for pattern, code in _ASCII_KEYWORD_PATTERNS:
        if pattern.search(lowered):
            return code
    return None


def map_ingredient_code(token: str) -> str | None:
    """재료명 → (선택) 알레르기 API 코드. 정확 일치·별칭만."""
    return map_allergy_code(token)


def resolve_canonical_allergen_label(token: str) -> str | None:
    """별칭/표준명이면 식약처 canonical 라벨, 아니면 None."""
    normalized = token.strip()
    if not normalized:
        return None
    alias_key = normalized.lower() if normalized.isascii() else normalized
    return ALIAS_TO_CANONICAL.get(normalized) or ALIAS_TO_CANONICAL.get(alias_key)


def _code_from_allergen_label(label: str) -> str | None:
    canonical = resolve_canonical_allergen_label(label) or label.strip()
    if not canonical:
        return None
    return _code_from_canonical_label(canonical) or map_allergy_code(canonical)


def resolve_ingredient_code(*, name: str, allergen: str | None) -> str | None:
    """Gemini allergen 표준명 우선, 없으면 재료명 별칭으로 코드 매핑."""
    if allergen:
        cleaned = allergen.strip()
        if cleaned and cleaned.lower() not in {"null", "none", "없음", "-"}:
            code = _code_from_allergen_label(cleaned)
            if code:
                return code
    return map_ingredient_code(name)


def parse_ingredient_item(raw: Any) -> tuple[str, str | None]:
    """ingredientsKo 항목(문자열 또는 {name, allergen}) → (표시명, 코드)."""
    allergen: str | None = None
    if isinstance(raw, dict):
        name = str(
            raw.get("name")
            or raw.get("ingredientName")
            or raw.get("재료")
            or ""
        ).strip()
        allergen_raw = raw.get("allergen")
        if allergen_raw is None:
            allergen_raw = raw.get("allergenName")
        if allergen_raw is not None:
            allergen = str(allergen_raw).strip() or None
    else:
        name = str(raw).strip()
    if not name:
        return "", None
    return name, resolve_ingredient_code(name=name, allergen=allergen)


def normalize_ingredient_name(token: str) -> tuple[str, str | None]:
    """문자열 재료명 → (원문 표시명, 선택적 코드)."""
    return parse_ingredient_item(token)


def format_mfds_labels_for_prompt() -> str:
    return ", ".join(MFDS_ALLERGEN_LABELS)


def build_ingredient_results(
    ingredients_ko: list[Any],
    *,
    base_confidence: float = 0.95,
    confidence_decay: float = 0.07,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """모델 재료 목록 → API ingredients (자유 이름 유지 + 선택적 코드)."""
    results: list[dict[str, Any]] = []
    for idx, raw in enumerate(ingredients_ko or []):
        name, code = parse_ingredient_item(raw)
        if not name:
            continue
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


def merge_allergy_results(
    *,
    allergens_ko: list[Any],
    ingredient_results: list[dict[str, Any]],
    fallback_confidence: float = 0.8,
) -> tuple[list[dict[str, Any]], list[str]]:
    """allergensKo를 기준으로 allergies를 만들고, 재료 코드를 합집합으로 보강."""
    allergies, unmapped = build_allergy_results(
        allergens_ko,
        fallback_confidence=fallback_confidence,
    )
    seen = {str(item.get("allergyCode") or "") for item in allergies}
    for item in ingredient_results:
        code = str(item.get("ingredientCode") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        allergies.append({"allergyCode": code, "confidence": fallback_confidence})
    return allergies, unmapped
