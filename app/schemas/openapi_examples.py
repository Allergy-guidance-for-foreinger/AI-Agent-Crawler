"""Swagger/OpenAPI용 요청·응답 예시 상수 (README와 동기화 권장)."""

from __future__ import annotations

# --- POST /api/v1/python/meals/crawl ---
MEAL_CRAWL_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "금오공대 일품식당 주간 조회",
        "description": "Accept-Language: ko 권장",
        "value": {
            "schoolName": "금오공과대학교",
            "cafeteriaName": "일품식당",
            "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
            "startDate": "2026-04-21",
            "endDate": "2026-04-27",
        },
    }
}

MEAL_CRAWL_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "schoolName": "금오공과대학교",
        "cafeteriaName": "일품식당",
        "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
        "startDate": "2026-04-21",
        "endDate": "2026-04-27",
        "meals": [
            {
                "mealDate": "2026-04-21",
                "mealType": "LUNCH",
                "menus": [
                    {"cornerName": "일품요리", "displayOrder": 1, "menuName": "김치찌개"},
                    {"cornerName": "일품요리", "displayOrder": 2, "menuName": "된장찌개"},
                ],
            }
        ],
    },
}

MEAL_CRAWL_ERROR_UPSTREAM_EXAMPLE: dict = {
    "success": False,
    "code": "PYM_502",
    "msg": "외부 크롤링 소스 조회에 실패했습니다. 잠시 후 다시 시도해주세요.",
}

MEAL_CRAWL_ERROR_BAD_CONDITION_EXAMPLE: dict = {
    "success": False,
    "code": "PYM_400",
    "msg": "요청 식단 조회 조건이 유효하지 않거나 데이터가 없습니다.",
}

# 식단 크롤 등에서 처리 중 예기치 않은 오류 시(문서용 예시; README PYM_500과 정합)
V1_INTERNAL_SERVER_ERROR_EXAMPLE: dict = {
    "success": False,
    "code": "PYM_500",
    "msg": "식단 조회 처리 중 서버 오류가 발생했습니다.",
}

# --- POST /api/v1/python/menus/analyze ---
MENU_ANALYZE_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "메뉴 1건 분석",
        "value": {"menus": [{"menuId": 101, "menuName": "김치찌개"}]},
    }
}

MENU_ANALYZE_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "results": [
            {
                "menuId": 101,
                "menuName": "김치찌개",
                "status": "SUCCESS",
                "spicyLevel": 3,
                "modelName": "gemini",
                "modelVersion": "gemini-2.5-flash",
                "ingredients": [
                    {"ingredientName": "김치", "ingredientCode": None, "confidence": 0.95},
                    {"ingredientName": "돼지고기", "ingredientCode": "PORK", "confidence": 0.88},
                    {"ingredientName": "두부", "ingredientCode": "SOYBEAN", "confidence": 0.81},
                ],
                "allergies": [
                    {"allergyCode": "SOYBEAN", "confidence": 0.85},
                    {"allergyCode": "PORK", "confidence": 0.8},
                ],
            }
        ]
    },
}

AI_KEY_MISSING_EXAMPLE: dict = {
    "success": False,
    "code": "AI_001",
    "msg": "GEMINI_API_KEY is not set",
}

# --- POST /api/v1/python/menus/describe ---
FOOD_DESCRIBE_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "김치찌개 설명",
        "value": {"menuId": 101, "menuName": "김치찌개"},
    }
}

FOOD_DESCRIBE_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "menuId": 101,
        "menuName": "김치찌개",
        "description": (
            "김치찌개는 잘 익은 김치와 돼지고기 또는 두부를 넣어 끓인 한국의 대표적인 찌개입니다. "
            "얼큰하고 새콤한 맛이 특징이며, 밥과 함께 자주 먹습니다."
        ),
        "modelName": "gemini",
        "modelVersion": "gemini-2.5-flash",
    },
}

# --- POST /api/v1/python/menus/describe/list ---
MENU_DESCRIBE_LIST_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "메뉴 2건 한국어 설명",
        "value": {
            "langCode": "ko",
            "menus": [
                {"menuId": 1, "menuName": "김치찌개"},
                {"menuId": 2, "menuName": "돈까스"},
            ],
        },
    }
}

MENU_DESCRIBE_LIST_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "results": [
            {"menuId": 1, "description": "김치와 돼지고기를 넣고 끓인 매콤한 한국식 찌개입니다."},
            {"menuId": 2, "description": "바삭하게 튀긴 돼지고기 커틀릿에 소스를 곁들인 음식입니다."},
        ]
    },
}

# --- POST /api/v1/python/menus/translate ---
MENU_TRANSLATE_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "영·일 번역",
        "value": {
            "menus": [{"menuId": 101, "menuName": "김치찌개"}],
            "targetLanguages": ["en", "ja"],
        },
    }
}

MENU_TRANSLATE_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "results": [
            {
                "menuId": 101,
                "sourceName": "김치찌개",
                "translations": [
                    {"langCode": "en", "translatedName": "Kimchi stew"},
                    {"langCode": "ja", "translatedName": "キムチチゲ"},
                ],
                "translationErrors": [],
            }
        ]
    },
}

# --- POST /api/v1/translations ---
FREE_TRANSLATION_REQUEST_OPENAPI_EXAMPLES: dict = {
    "기본": {
        "summary": "한→영 문장 번역",
        "value": {
            "sourceLang": "ko",
            "targetLang": "en",
            "text": "이 음식에 밀가루가 들어가나요?",
        },
    }
}

FREE_TRANSLATION_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "sourceLang": "ko",
        "targetLang": "en",
        "text": "이 음식에 밀가루가 들어가나요?",
        "translatedText": "Does this dish contain flour?",
    },
}

# --- POST /api/v1/python/translations/list ---
TEXT_LIST_TRANSLATION_REQUEST_OPENAPI_EXAMPLES: dict = {
    "재료목록": {
        "summary": "재료명 ko → en",
        "value": {
            "sourceLang": "ko",
            "targetLang": "en",
            "ingredients": [
                {"ingredientCode": "AI_AB12CD34", "text": "베이컨"},
                {"ingredientCode": "AI_CD34EF56", "text": "소고기 패티"},
            ],
        },
    }
}

TEXT_LIST_TRANSLATION_SUCCESS_EXAMPLE: dict = {
    "success": True,
    "data": {
        "results": [
            {"ingredientCode": "AI_AB12CD34", "translatedText": "Bacon"},
            {"ingredientCode": "AI_CD34EF56", "translatedText": "Beef patty"},
        ]
    },
}

# --- multipart (문서용 설명; Swagger는 폼 필드 설명으로 표시) ---
MENU_BOARD_ANALYZE_RESPONSE_EXAMPLE: dict = {
    "success": True,
    "data": {
        "requestId": "req-001",
        "recognizedMenus": [{"menuName": "김치찌개", "confidence": 0.82}],
    },
}

FOOD_IMAGE_ANALYZE_RESPONSE_EXAMPLE: dict = {
    "success": True,
    "data": {
        "requestId": "req-002",
        "foodName": "김치찌개",
        "ingredients": [
            {"ingredientName": "두부", "ingredientCode": "SOYBEAN", "confidence": 0.9},
            {"ingredientName": "밀가루", "ingredientCode": "WHEAT", "confidence": 0.75},
        ],
        "notes": "추정 결과이며 실제와 다를 수 있습니다.",
    },
}

VALIDATION_ERROR_EXAMPLE: dict = {
    "success": False,
    "code": "COM_002",
    "msg": "요청 데이터 변환 과정에서 오류가 발생했습니다.",
}

BAD_REQUEST_COM001_EXAMPLE: dict = {
    "success": False,
    "code": "COM_001",
    "msg": "이미지 파일이 비어 있습니다.",
}

PAYLOAD_TOO_LARGE_COM001_EXAMPLE: dict = {
    "success": False,
    "code": "COM_001",
    "msg": "이미지 파일이 너무 큽니다 (최대 25MB).",
}
