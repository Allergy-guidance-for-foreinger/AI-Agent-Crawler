# AI-Agent-Crawler 계층 구조

코드 탐색 기준은 아래와 같습니다.

## 계층 가이드

- `app/api/routes`
  - FastAPI 라우터와 HTTP 입출력 처리
  - `live.py`: 레거시 + `/api/v1/python/*` (래핑 응답)
  - `spring_native.py`: Spring `MealCrawlProperties` 기본 경로 (`/api/v1/crawl/meals` 등, 비래핑)
- `app/services`
  - 유스케이스 오케스트레이션(`live_service.py`), 순수 로직(`ops.py`)
  - 식단 크롤은 `crawl_daily_meals()`가 `sourceUrl` 호스트로 학교 어댑터를 선택합니다.
- `app/repositories`
  - 외부 I/O 오케스트레이션(Spring HTTP, Gemini 호출, 크롤 유스케이스 위임)
- `app/domain`
  - 도메인 모델 및 정책
  - 학교별 크롤러 어댑터(`crawler/kumoh_menu.py`, `crawler/knu_menu.py`)는 소스 HTML fetch를 포함할 수 있습니다.
- `app/schemas`
  - 요청/응답 Pydantic 모델 및 OpenAPI 예시
- `app/config`
  - 런타임 설정 및 앱 팩토리
- `app/common`
  - 공통 유틸 및 `ops` 재노출

## 호환성 정책

- `user_features/live_service.py`는 그대로 두며, 루트 `main.py`는 동일 앱을 노출합니다.
- 신규 개발은 `app/*` 위 구조를 따릅니다.

## 실행 엔트리

- API 서버: `python3 -m uvicorn main:app --host 0.0.0.0 --port 8000`
- API 서버(대안): `python3 -m uvicorn user_features.live_service:app --host 0.0.0.0 --port 8000`
- CLI: `scripts/` 하위 파일 사용 권장
