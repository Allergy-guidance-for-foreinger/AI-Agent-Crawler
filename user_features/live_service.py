"""상시 구동 서비스 엔트리포인트.

이 파일은 app 조립만 담당합니다.
실제 비즈니스 로직은 `user_features/live/` 하위 모듈로 분리했습니다.
"""

from __future__ import annotations

import repo_env
from app.bootstrap.app_factory import create_app
from app.bootstrap.runtime import load_runtime_context

# .env 로딩은 서비스 시작 시 단 한 번 실행합니다.
repo_env.load_dotenv_from_repo_root()
RUNTIME = load_runtime_context()
app = create_app(RUNTIME)
