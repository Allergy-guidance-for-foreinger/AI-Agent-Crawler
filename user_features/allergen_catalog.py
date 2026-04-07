"""식약처 표시 대상 알레르기(한국어) 기준 별칭·확장 키워드."""

from __future__ import annotations

# 공식 표기(한글) → 요약 문자열/모델 출력에 자주 나오는 동의어·포함 식품 힌트
EXPANSIONS: dict[str, frozenset[str]] = {
    "난류": frozenset({"난류", "계란", "달걀", "에그", "유정란", "난황", "알부민"}),
    "우유": frozenset(
        {
            "우유",
            "치즈",
            "요거트",
            "요플레",
            "요구르트",
            "버터",
            "크림",
            "유당",
            "카제인",
            "연유",
        }
    ),
    "메밀": frozenset({"메밀", "메밀가루"}),
    "땅콩": frozenset({"땅콩", "피넛"}),
    "대두": frozenset({"대두", "두부", "된장", "간장", "콩", "순두부", "유부"}),
    "밀": frozenset({"밀", "밀가루", "밀류", "글루텐", "튀김옷", "빵"}),
    "고등어": frozenset({"고등어"}),
    "게": frozenset({"게"}),
    "새우": frozenset({"새우", "새우류"}),
    "돼지고기": frozenset({"돼지", "돼지고기", "한돈", "제육", "돈까스", "너비아니"}),
    "복숭아": frozenset({"복숭아", "복숭아과"}),
    "토마토": frozenset({"토마토"}),
    "아황산류": frozenset({"아황산", "아황산류", "이황산"}),
    "호두": frozenset({"호두"}),
    "닭고기": frozenset({"닭", "닭고기", "치킨"}),
    "쇠고기": frozenset({"쇠고기", "소고기", "소고", "육우"}),
    "오징어": frozenset({"오징어", "한치", "무늬오징어"}),
    "조개류": frozenset({"조개", "조개류", "굴", "홍합", "전복", "가리비"}),
    "잣": frozenset({"잣"}),
}

# 사용자 입력(한·영·구어) → 위 canonical 키
ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _syns in EXPANSIONS.items():
    ALIAS_TO_CANONICAL[_canon] = _canon
    for _s in _syns:
        ALIAS_TO_CANONICAL[_s] = _canon

ALIAS_TO_CANONICAL.update(
    {
        "egg": "난류",
        "eggs": "난류",
        "milk": "우유",
        "peanut": "땅콩",
        "peanuts": "땅콩",
        "soy": "대두",
        "soybean": "대두",
        "wheat": "밀",
        "gluten": "밀",
        "shrimp": "새우",
        "pork": "돼지고기",
        "beef": "쇠고기",
        "chicken": "닭고기",
        "walnut": "호두",
        "pine nut": "잣",
        "pinenut": "잣",
    }
)


def normalize_user_allergen_tokens(raw: list[str]) -> set[str]:
    """쉼표 등으로 나뉜 사용자 입력을 canonical 집합으로."""
    out: set[str] = set()
    for item in raw:
        k = item.strip()
        if not k:
            continue
        key = k.lower() if k.isascii() else k
        canon = ALIAS_TO_CANONICAL.get(k) or ALIAS_TO_CANONICAL.get(key)
        if canon:
            out.add(canon)
        else:
            out.add(k)
    return out


def list_canonical_choices() -> list[str]:
    return sorted(EXPANSIONS.keys())
