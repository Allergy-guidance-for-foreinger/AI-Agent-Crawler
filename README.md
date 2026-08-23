# AI-Agent-Crawler

Spring Boot에서 호출하는 **Python API 서버**입니다. 연동 시에는 Swagger 태그 **`실사용 API`** 에 해당하는 엔드포인트만 쓰면 됩니다. OCR·메뉴 일괄 번역 등 **나머지 API는 필요할 때만** 사용하면 되고, 안 써도 서비스 동작에는 영향 없습니다.

## 실사용 API (연동 권장)

| 메서드 | 경로 | 응답 | 용도 |
|---|---|---|---|
| `POST` | `/api/v1/python/meals/crawl` | Wrapped | 주간 식단 크롤링 |
| `POST` | `/api/v1/python/menus/analyze` | Wrapped | 메뉴 재료·알레르기·매운맛(0~5) 분석 |
| `POST` | `/api/v1/python/menus/analyze-image` | Wrapped | 음식 사진 → 한국어명·번역·발음·근거·confidence |
| `POST` | `/api/v1/python/menus/describe` | Wrapped | menuId·menuName → 한국어 음식 설명 |
| `POST` | `/api/v1/python/menus/describe/list` | Wrapped | langCode + 메뉴 목록 → 메뉴별 설명 목록 |
| `POST` | `/api/v1/translations` | Unwrapped | 자유 텍스트 번역 (질문/안내 문구 등) |
| `POST` | `/api/v1/python/translations/list` | Wrapped | 재료명 등 문자열 **목록** 일괄 번역 |
| `POST` | `/api/v1/translations/list` | Unwrapped | 동일 (응답 본문이 `results` 객체) |

- **Wrapped**: `{ "success": true, "data": { ... } }` 형태 (`/api/v1/python/...`)
- **Unwrapped**: 본문에 결과만 반환 (`/api/v1/translations`)

Spring `WebClient`가 DTO를 바로 역직렬화하기 편하면, 아래 **동일 로직·Unwrapped** 경로를 대신 써도 됩니다.

| 실사용 (Wrapped) | 동일 기능 (Unwrapped) |
|---|---|
| `POST /api/v1/python/meals/crawl` | `POST /api/v1/crawl/meals` |
| `POST /api/v1/python/menus/analyze` | `POST /api/v1/menus/analyze` |
| `POST /api/v1/python/menus/describe` | `POST /api/v1/menus/describe` |
| `POST /api/v1/python/menus/describe/list` | `POST /api/v1/menus/describe/list` |
| `POST /api/v1/python/translations/list` | `POST /api/v1/translations/list` |

헬스(래핑 없음, `/api/v1` 밖): `GET /health`

## 선택 API (필요 시만)

아래는 프로토타입·보조 기능용입니다. **사용하지 않아도 됩니다.**

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/api/v1/python/menus/ocr` | 메뉴판 이미지 OCR |
| `POST` | `/api/v1/python/menus/analyze-from-ocr` | OCR 후 바로 메뉴 분석 |
| `POST` | `/api/v1/python/menus/translate` | 저장된 메뉴명 일괄 번역 (배치) |

Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — **`실사용 API`** 그룹부터 확인하면 됩니다.

---

## 요구 사항

- Python 3.10+
- `pip install -r requirements.txt`
- AI 분석/번역 사용 시 `GEMINI_API_KEY` 필요

---

## 실행

### 로컬 (직접 실행)

```bash
pip install -r requirements.txt
cp .env.example .env
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 등 필요한 값 설정

# docker compose로 실행
docker compose up -d

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

빌드만 따로 하려면:

```bash
docker build -t ai-agent-crawler .
docker run -d --env-file .env -p 8000:8000 ai-agent-crawler
```

문서 확인 (서버 실행 후 **http** 로 접속):

- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- OpenAPI: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 공통 규칙

- Base URL: `/api/v1`
- 헤더:
  - `Content-Type`: 기본은 `application/json`, 이미지 업로드 API는 `multipart/form-data`
  - `Accept-Language: ko | en | zh-CN | vi | ja` (`en-US`, `ko-KR` 같은 locale 변형도 허용)

### Wrapped 응답 (`/api/v1/python/...`)

```json
{
  "success": true,
  "data": {}
}
```

### Unwrapped 응답 (Spring Native)

래핑 없이 결과를 직접 반환합니다.

```json
{
  "schoolName": "...",
  "cafeteriaName": "...",
  "meals": [...]
}
```

### 실패 응답 (공통)

```json
{
  "success": false,
  "code": "COM_002",
  "msg": "요청 데이터 변환 과정에서 오류가 발생했습니다."
}
```

---

## 헬스

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 프로세스 헬스 (`{"status":"ok"}`, `/api/v1` 밖) |

---

## 실사용 API 상세

### 1) 식단 크롤링

### `POST /api/v1/python/meals/crawl`

Java 서버 스케줄러가 호출하여 일주일치 식단을 수집할 때 사용합니다.

### 요청 DTO

```java
public record PythonMealCrawlRequest(
        String schoolName,
        String cafeteriaName,
        String sourceUrl,
        LocalDate startDate,
        LocalDate endDate
) { }
```

### 지원 학교·식당·sourceUrl

`sourceUrl` 호스트로 학교 어댑터를 선택합니다. `cafeteriaName`은 아래 표와 일치해야 합니다(금오는 구명칭 학생식당·교직원식당을 자동 치환).

경북대·경국대는 **쿼리 파라미터만 있으면** 경로를 생략해도 됩니다. 서버가 조회용 정규 경로로 보정합니다.
예: `https://coop.knu.ac.kr/?shop_sqno=35`, `https://www.gknu.ac.kr/?menu_idx=82`

#### 금오공과대학교 (`www.kumoh.ac.kr`)

| cafeteriaName | sourceUrl |
|---|---|
| 일품식당 | `https://www.kumoh.ac.kr/ko/restaurant01.do` |
| 정찬식당 | `https://www.kumoh.ac.kr/ko/restaurant02.do` |
| 분식당 | `https://www.kumoh.ac.kr/ko/restaurant04.do` |

#### 경북대학교 (`coop.knu.ac.kr`)

| cafeteriaName | sourceUrl |
|---|---|
| 정보센터식당 | `https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35` |
| 복지관 교직원식당 | `https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=36` |
| 카페테리아 첨성 | `https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=37` |
| GP감꽃식당 | `https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=46` |
| 공학관교직원식당(외부업체) | `https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=85` |

#### 국립경국대학교 (`www.gknu.ac.kr`)

| cafeteriaName | sourceUrl |
|---|---|
| 이룸관(안동, 학생식당) | `https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82` |
| 채움관(안동, 교직원식당) | `https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=222` |
| 양식코너(안동) | `https://www.gknu.ac.kr/main/html.do?menu_idx=317` |
| 학생식당(예천) | `https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=629` |

### 응답 DTO

```java
public record PythonMealCrawlResponse(
        String schoolName,
        String cafeteriaName,
        String sourceUrl,
        LocalDate startDate,
        LocalDate endDate,
        List<PythonDailyMealDto> meals
) { }

public record PythonDailyMealDto(
        LocalDate mealDate,
        String mealType,
        List<PythonCrawledMenuDto> menus
) { }

public record PythonCrawledMenuDto(
        String cornerName,
        Integer displayOrder,
        String menuName
) { }
```

### 학교별 호출·응답 예시

아래는 `POST /api/v1/python/meals/crawl` 기준입니다. Unwrapped(`POST /api/v1/crawl/meals`)는 응답에서 `success`/`data` 래핑 없이 `data` 본문만 반환합니다. `meals` 일부만 축약해 표기했습니다.

#### 금오공과대학교

요청:

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "일품식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
  "startDate": "2026-04-15",
  "endDate": "2026-04-21"
}
```

응답:

```json
{
  "success": true,
  "data": {
    "schoolName": "금오공과대학교",
    "cafeteriaName": "일품식당",
    "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
    "startDate": "2026-04-15",
    "endDate": "2026-04-21",
    "meals": [
      {
        "mealDate": "2026-04-15",
        "mealType": "BREAKFAST",
        "menus": [
          { "cornerName": "조식", "displayOrder": 1, "menuName": "다찬스페셜정식도시락" }
        ]
      },
      {
        "mealDate": "2026-04-15",
        "mealType": "LUNCH",
        "menus": [
          { "cornerName": "일품요리", "displayOrder": 1, "menuName": "김치우동" },
          { "cornerName": "일품요리", "displayOrder": 2, "menuName": "목살필라프" },
          { "cornerName": "일품요리", "displayOrder": 3, "menuName": "참치마요덮밥" }
        ]
      }
    ]
  }
}
```

#### 경북대학교

요청:

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "정보센터식당",
  "sourceUrl": "https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
  "startDate": "2026-07-27",
  "endDate": "2026-07-27"
}
```

응답:

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "정보센터식당",
    "sourceUrl": "https://coop.knu.ac.kr/sub03/sub01_01.html?shop_sqno=35",
    "startDate": "2026-07-27",
    "endDate": "2026-07-27",
    "meals": [
      {
        "mealDate": "2026-07-27",
        "mealType": "LUNCH",
        "menus": [
          { "cornerName": "특식", "displayOrder": 1, "menuName": "육전비빔국수★ 망고쥬스" },
          { "cornerName": "특식", "displayOrder": 2, "menuName": "동파육볶음밥★ 군만두" },
          { "cornerName": "특식", "displayOrder": 3, "menuName": "불고기비빔밥★" }
        ]
      },
      {
        "mealDate": "2026-07-27",
        "mealType": "DINNER",
        "menus": [
          { "cornerName": "특식", "displayOrder": 1, "menuName": "촌돼지찌개★" },
          { "cornerName": "특식", "displayOrder": 2, "menuName": "오삼불고기덮밥★" },
          { "cornerName": "특식", "displayOrder": 3, "menuName": "순살돈가스★" }
        ]
      }
    ]
  }
}
```

#### 국립경국대학교

요청:

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "이룸관(안동, 학생식당)",
  "sourceUrl": "https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
  "startDate": "2026-07-24",
  "endDate": "2026-07-24"
}
```

응답:

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "이룸관(안동, 학생식당)",
    "sourceUrl": "https://www.gknu.ac.kr/main/module/foodMenu/index.do?menu_idx=82",
    "startDate": "2026-07-24",
    "endDate": "2026-07-24",
    "meals": [
      {
        "mealDate": "2026-07-24",
        "mealType": "BREAKFAST",
        "menus": [
          { "cornerName": "조식", "displayOrder": 1, "menuName": "[천원의 아침밥]" }
        ]
      },
      {
        "mealDate": "2026-07-24",
        "mealType": "LUNCH",
        "menus": [
          { "cornerName": "중식", "displayOrder": 1, "menuName": "[천원의 브런치]" },
          { "cornerName": "중식", "displayOrder": 2, "menuName": "흑미밥" },
          { "cornerName": "중식", "displayOrder": 3, "menuName": "쇠고기미역국" },
          { "cornerName": "중식", "displayOrder": 4, "menuName": "꿔바로우" }
        ]
      }
    ]
  }
}
```

> **메뉴 분리 규칙 (금오)**: HTML `<li>` 태그 기준으로 메뉴를 개별 분리합니다.
> 셀에 포함된 시간 범위(`11:00~14:00`), 메타정보(`[천원의 아침밥]`), 안내문(`*재학생만 해당`) 등은 자동 필터링됩니다.
> `cornerName`은 셀의 첫 번째 항목(조식, 일품요리, 중식 등)에서 추출되며, `mealType`도 이로부터 추론됩니다.
>
> **메뉴 분리 규칙 (경북대)**: `week_table`의 `ul.menu_im > li`(가격 단위)로 메뉴를 분리합니다.
> `cornerName`은 셀의 `특식`/`정식` 배지에서 가져오고, `mealType`은 중식/석식 테이블 라벨로 결정합니다.
> 기간이 여러 날에 걸치면 `selDate`(페이지 첫 칸 날짜)를 6일 간격으로 조회해 병합합니다.
> `selDate`는 월요일이 아니라 요청 `startDate`부터 시작합니다.
>
> **메뉴 분리 규칙 (경국대)**: 일별 AJAX(`foodMenu/view.do`)의 `dl/dt/dd`에서 조식·중식·석식을 파싱하고 `<br>` 단위로 메뉴를 분리합니다.
> 양식코너는 고정 가격표 HTML을 1회 조회한 뒤 요청 기간의 평일(월~금)에만 `LUNCH`로 동일 메뉴를 반복합니다(주말 제외).
> `미 운 영`·빈 칸은 제외합니다.

실패 응답 예시:

```json
{ "success": false, "code": "PYM_400", "msg": "요청 식단 조회 조건이 유효하지 않거나 데이터가 없습니다." }
```

```json
{ "success": false, "code": "PYM_502", "msg": "외부 크롤링 소스 조회에 실패했습니다. 잠시 후 다시 시도해주세요." }
```

---

동일 로직 Unwrapped: `POST /api/v1/crawl/meals` — 요청/응답 본문은 `data` 없이 위 `data` 내용과 동일합니다.

---

### 2) 메뉴 분석

#### `POST /api/v1/python/menus/analyze`

식단 저장 후 분석이 없는 메뉴만 Java 서버가 요청합니다. 응답 스키마는 코드 기준 `app/schemas/api_models.py`의 `PythonMenuAnalysisResultDto`와 동일합니다.

### 요청 DTO

```java
public record PythonMenuAnalysisRequest(
        List<PythonMenuAnalysisTargetDto> menus
) { }

public record PythonMenuAnalysisTargetDto(
        Long menuId,
        String menuName
) { }
```

요청 예시:

```json
{
  "menus": [
    { "menuId": 101, "menuName": "김치찌개" },
    { "menuId": 102, "menuName": "돈까스" }
  ]
}
```

### 응답 DTO

```java
public record PythonMenuAnalysisResponse(
        List<PythonMenuAnalysisResultDto> results
) { }

public record PythonMenuAnalysisResultDto(
        Long menuId,
        String menuName,
        PythonMenuAnalysisStatus status,
        Long spicyLevel,
        String modelName,
        String modelVersion,
        List<PythonMenuIngredientResultDto> ingredients,
        List<PythonMenuAllergyResultDto> allergies
) { }

public record PythonMenuIngredientResultDto(
        String ingredientName,
        String ingredientCode,
        BigDecimal confidence
) { }

public record PythonMenuAllergyResultDto(
        String allergyCode,
        BigDecimal confidence
) { }
```

- `status`: 성공 시 `SUCCESS`, 예외 시 `FAILED`.
- `spicyLevel`: 매운맛 **0~5** 정수 또는 `null`. **0**=매운맛 없음(밥 등), **null**=실패·미추정.
- `ingredients`: 추정 재료 목록(`ingredientName` 필수, `ingredientCode`는 매핑 성공 시만).
  재료명은 자유 텍스트이며, Gemini `allergen` 표준명 또는 별칭으로 코드가 붙을 수 있습니다.
- `allergies`: 알레르기 코드 목록(`allergyCode`). `allergensKo` 표준명으로 확정하고, 재료 코드로 보강합니다. 분석 실패 시 빈 배열.

성공 응답 예시:

```json
{
  "success": true,
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
          { "ingredientName": "김치", "ingredientCode": null, "confidence": 0.95 },
          { "ingredientName": "돼지고기", "ingredientCode": "PORK", "confidence": 0.88 },
          { "ingredientName": "간장", "ingredientCode": "SOYBEAN", "confidence": 0.81 }
        ],
        "allergies": [
          { "allergyCode": "PORK", "confidence": 0.8 },
          { "allergyCode": "SOYBEAN", "confidence": 0.8 }
        ],
        "spicyLevel": 3
      }
    ]
  }
}
```

실패 응답 예시:

```json
{ "success": false, "code": "AI_001", "msg": "GEMINI_API_KEY is not set" }
```

동일 로직 Unwrapped: `POST /api/v1/menus/analyze` — 응답은 `{ "results": [ ... ] }` 형태이며, 하위 호환용 `spicy_level`(snake_case)가 `spicyLevel`과 같이 올 수 있습니다.

---

### 3) 메뉴명 한국어 설명

#### `POST /api/v1/python/menus/describe`

`menuId`·`menuName`을 주면 Gemini가 해당 음식을 **한국어로 2~4문장** 설명합니다. (재료·알레르기 분석과 별도 API)

요청 예시:

```json
{
  "menuId": 101,
  "menuName": "김치찌개"
}
```

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "menuId": 101,
    "menuName": "김치찌개",
    "description": "김치찌개는 잘 익은 김치와 돼지고기 또는 두부를 넣어 끓인 한국의 대표적인 찌개입니다. 얼큰하고 새콤한 맛이 특징입니다.",
    "modelName": "gemini",
    "modelVersion": "gemini-2.5-flash"
  }
}
```

동일 로직 Unwrapped: `POST /api/v1/menus/describe` — `success`/`data` 없이 위 `data` 객체를 그대로 반환합니다.

#### `POST /api/v1/python/menus/describe/list`

여러 메뉴를 한 번에 설명받는 API입니다. 요청 형식은 아래와 같습니다.

```json
{
  "langCode": "ko",
  "menus": [
    { "menuId": 1, "menuName": "김치찌개" },
    { "menuId": 2, "menuName": "돈까스" }
  ]
}
```

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "results": [
      { "menuId": 1, "description": "김치와 돼지고기를 넣고 끓인 매콤한 한국식 찌개입니다." },
      { "menuId": 2, "description": "바삭하게 튀긴 돼지고기 커틀릿에 소스를 곁들인 음식입니다." }
    ]
  }
}
```

동일 로직 Unwrapped: `POST /api/v1/menus/describe/list` — 응답은 `{ "results": [ ... ] }` 형태입니다.

---

### 4) 자유 텍스트 번역

#### `POST /api/v1/translations`

UI 문구·챗봇 질문 등 **임의 텍스트**를 번역할 때 사용합니다. (메뉴 DB 일괄 번역은 선택 API `POST /api/v1/python/menus/translate` 참고)

**요청:**

```json
{
  "sourceLang": "ko",
  "targetLang": "en",
  "text": "이 메뉴에 땅콩이 들어가나요?"
}
```

**응답 (200, Unwrapped):**

```json
{
  "translatedText": "Does this menu contain peanuts?"
}
```

#### `POST /api/v1/translations/list` (재료 목록)

`ingredientCode`가 붙은 재료 목록을 한 번에 번역합니다.

**요청:**

```json
{
  "sourceLang": "ko",
  "targetLang": "en",
  "ingredients": [
    { "ingredientCode": "AI_AB12CD34", "text": "베이컨" },
    { "ingredientCode": "AI_CD34EF56", "text": "소고기 패티" }
  ]
}
```

**응답 (200, Unwrapped):**

```json
{
  "results": [
    { "ingredientCode": "AI_AB12CD34", "translatedText": "Bacon" },
    { "ingredientCode": "AI_CD34EF56", "translatedText": "Beef patty" }
  ]
}
```

Wrapped (`POST /api/v1/python/translations/list`):

```json
{
  "success": true,
  "data": {
    "results": [
      { "ingredientCode": "AI_AB12CD34", "translatedText": "Bacon" },
      { "ingredientCode": "AI_CD34EF56", "translatedText": "Beef patty" }
    ]
  }
}
```

---

### 5) 음식 이미지 분석

#### `POST /api/v1/python/menus/analyze-image`

음식 사진에서 **한국어 음식명**, **요청 `language` 번역**, **발음 표기**, **판단 근거(요청 language)**, **confidence** 를 반환합니다.

요청 형식:

- `multipart/form-data`
- `language`: `ko | en | zh-CN | vi | ja` (기본값 `ko`)
- `image`: (필수)

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "identifiedFoodKoreanName": "제육볶음",
        "identifiedFoodTranslationName": "Spicy stir-fried pork",
        "identifiedFoodPronunciationName": "je-yuk-bokkeum",
        "identifiedFoodNameReason": "The dish shows stir-fried pork in red chili sauce, similar to Korean spicy pork.",
        "confidence": 0.81,
        "modelName": "gemini",
        "modelVersion": "gemini-2.5-flash"
      }
    ]
  }
}
```

`language=ko`이면 `identifiedFoodTranslationName`, `identifiedFoodPronunciationName`, `identifiedFoodNameReason` 모두 한국어로 반환됩니다.

실패 응답 예시:

```json
{ "success": false, "code": "COM_001", "msg": "이미지 파일이 비어 있습니다." }
```

```json
{ "success": false, "code": "AI_001", "msg": "GEMINI_API_KEY is not set" }
```

---

## 선택 API 상세

### 메뉴판 OCR

#### `POST /api/v1/python/menus/ocr`

메뉴판 이미지에서 OCR 방식으로 텍스트를 읽고 메뉴 목록을 추출합니다.

요청 형식:

- `multipart/form-data`
- `image`: 메뉴판 이미지 파일 (필수)

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "rawText": "중식\n김치찌개\n돈까스\n비빔밥",
    "menus": [
      { "menuName": "김치찌개" },
      { "menuName": "돈까스" },
      { "menuName": "비빔밥" }
    ]
  }
}
```

---

### 메뉴판 OCR + 분석

#### `POST /api/v1/python/menus/analyze-from-ocr`

메뉴판 OCR 결과를 바로 메뉴 분석으로 연결합니다.

요청 형식:

- `multipart/form-data`
- `image`: 메뉴판 이미지 파일 (필수)
- `startMenuId`: 응답 `menuId` 시작값 (선택, 기본값 `1`)

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "menuId": 1,
        "menuName": "김치찌개",
        "status": "SUCCESS",
        "modelName": "gemini",
        "modelVersion": "gemini-2.5-flash",
        "analyzedAt": "2026-04-15T09:30:00",
        "ingredients": [
          { "ingredientCode": "PORK", "confidence": 0.92 }
        ],
        "allergies": [],
        "spicyLevel": 2
      }
    ]
  }
}
```

---

### 메뉴 일괄 번역 (배치)

#### `POST /api/v1/python/menus/translate`

저장된 메뉴 목록을 언어별로 번역할 때만 사용합니다. 자유 문장 번역은 실사용 API `POST /api/v1/translations` 를 사용하세요.

요청 예시:

```json
{
  "menus": [
    { "menuId": 101, "menuName": "김치찌개" },
    { "menuId": 102, "menuName": "돈까스" }
  ],
  "targetLanguages": ["en"]
}
```

성공 응답 예시:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "menuId": 101,
        "sourceName": "김치찌개",
        "translations": [
          { "langCode": "en", "translatedName": "Kimchi Stew" }
        ]
      }
    ]
  }
}
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | 없음 | 메뉴 분석/번역에 필요 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | AI 모델명 |
| `AI_MAX_CONCURRENT_TASKS` | `4` | 분석/번역 동시성 |
| `SERVICE_TIMEZONE` | `Asia/Seoul` | 분석 시각 타임존 |
| `CRAWL_SOURCE_ALLOWLIST` | 없음(제한 없음) | 크롤 허용 호스트 화이트리스트. 설정 시 해당 호스트만 `sourceUrl` 허용 (SSRF 방어) |

예시:

```env
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash
AI_MAX_CONCURRENT_TASKS=4
SERVICE_TIMEZONE=Asia/Seoul
CRAWL_SOURCE_ALLOWLIST=www.kumoh.ac.kr,kumoh.ac.kr
```

---

## 테스트

```bash
python3 -m pytest -q tests/live                # 단위 테스트 (메뉴 분리, 서비스, 리포지토리)
python3 -m pytest -q tests/integration          # 통합 테스트 (AI API, Spring 계약)
```

---

## AWS 배포 가이드 (친구가 README만 보고 배포)

이 서비스는 Dockerfile 없이도 배포 가능합니다. 현재 저장소 기준으로는 **EC2 + systemd + Nginx + ACM/Certbot** 구성이 가장 빠르고 안정적입니다.

### 권장 아키텍처

- **EC2 (Ubuntu 22.04)**: FastAPI(Uvicorn) 실행
- **systemd**: 프로세스 자동 재시작/부팅 시 자동 실행
- **Nginx**: 80/443 리버스 프록시
- **Route53 + 인증서**: 도메인/TLS
- **보안그룹**:
  - 인바운드: `22`(관리자 IP 제한), `80`, `443`
  - 아웃바운드: 기본 허용 (Gemini, 외부 메뉴 URL 호출 필요)

### 1) EC2 서버 준비

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git
```

애플리케이션용 계정/디렉터리:

```bash
sudo mkdir -p /opt/ai-agent-crawler
sudo chown -R $USER:$USER /opt/ai-agent-crawler
cd /opt/ai-agent-crawler
```

소스 배포:

```bash
git clone <REPO_URL> .
git checkout DTO-확정
```

### 2) Python 런타임 설치

```bash
cd /opt/ai-agent-crawler
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) 운영 환경변수 설정 (`.env`)

```bash
cd /opt/ai-agent-crawler
cp .env.example .env
```

최소 필수(운영 권장):

```env
GEMINI_API_KEY=YOUR_GEMINI_KEY
GEMINI_MODEL=gemini-2.5-flash
SERVICE_TIMEZONE=Asia/Seoul
AI_MAX_CONCURRENT_TASKS=4
# SSRF 방어 권장
CRAWL_SOURCE_ALLOWLIST=www.kumoh.ac.kr,kumoh.ac.kr
```

### 4) systemd 서비스 등록

`/etc/systemd/system/ai-agent-crawler.service` 생성:

```ini
[Unit]
Description=AI Agent Crawler FastAPI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ai-agent-crawler
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/opt/ai-agent-crawler/.env
ExecStart=/opt/ai-agent-crawler/.venv/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

적용:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-agent-crawler
sudo systemctl start ai-agent-crawler
sudo systemctl status ai-agent-crawler
```

로그 확인:

```bash
journalctl -u ai-agent-crawler -f
```

### 5) Nginx 리버스 프록시 설정

`/etc/nginx/sites-available/ai-agent-crawler`:

```nginx
server {
    listen 80;
    server_name api.your-domain.com;

    # 앱 이미지 한도 25MB; multipart 오버헤드용으로 30M 권장
    client_max_body_size 30M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

활성화:

```bash
sudo ln -s /etc/nginx/sites-available/ai-agent-crawler /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6) HTTPS 적용 (권장)

옵션 A: Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.your-domain.com
```

옵션 B: ALB + ACM(팀 운영에 더 적합)

- ALB(443)에서 ACM 인증서 연결
- Target Group은 EC2:80 또는 EC2:8000
- 헬스체크는 `/docs` 또는 `/openapi.json` 권장

### 7) 배포 후 체크리스트

서버 내부 헬스 체크:

```bash
curl -sS http://127.0.0.1:8000/openapi.json | jq .openapi
```

외부 도메인 체크:

```bash
curl -sS https://api.your-domain.com/docs
```

샘플 API 호출:

```bash
curl -sS -X POST "https://api.your-domain.com/api/v1/python/menus/analyze" \
  -H "Content-Type: application/json" \
  -H "Accept-Language: ko" \
  -d '{"menus":[{"menuId":1,"menuName":"김치찌개"}]}'
```

### 8) 운영 시 반드시 확인할 설정값

| 키 | 필수 여부 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | 필수 | 분석/번역/OCR 모든 AI 기능에 필요 |
| `GEMINI_MODEL` | 권장 | 기본값 사용 가능하나 운영에서 고정 권장 |
| `SERVICE_TIMEZONE` | 권장 | `analyzedAt` 생성 타임존 |
| `AI_MAX_CONCURRENT_TASKS` | 권장 | 동시 AI 호출 수 |
| `CRAWL_SOURCE_ALLOWLIST` | 강력 권장 | 크롤링 `sourceUrl` 호스트 제한 (SSRF 방어) |
| `SPRING_API_TOKEN` | 선택 | Spring 내부 API 보호 시 Bearer 토큰 |
| `SPRING_API_KEY` | 선택 | Spring 내부 API가 X-API-Key 요구 시 |

### 9) 장애 대응 포인트

- `502/504` 다발: 외부 메뉴 사이트 응답 지연 가능성, `sourceUrl` 접근성 점검
- `AI_001` 응답: `GEMINI_API_KEY` 누락/오타
- OCR 결과 빈 값: 업로드 이미지 품질/해상도 확인, 메뉴판 crop 후 재시도
- 크롤링 차단: `CRAWL_SOURCE_ALLOWLIST` 설정값과 실제 도메인 일치 확인

