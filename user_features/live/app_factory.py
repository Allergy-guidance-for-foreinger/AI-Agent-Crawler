"""FastAPI app 조립 모듈."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from user_features.live.routers import create_legacy_router, create_v1_router
from user_features.live.runtime import API_V1_PREFIX, RuntimeContext
from user_features.live.service_ops import next_run, run_weekly_crawl_once, v1_error

logger = logging.getLogger(__name__)


def create_app(ctx: RuntimeContext) -> FastAPI:
    """애플리케이션 인스턴스를 조립해서 반환한다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async def _weekly_loop() -> None:
            tz = ZoneInfo(ctx.config.timezone_name)
            while True:
                now = datetime.now(tz)
                target = next_run(
                    now,
                    weekday=ctx.config.crawl_weekday,
                    hour=ctx.config.crawl_hour,
                    minute=ctx.config.crawl_minute,
                )
                await asyncio.sleep(max((target - now).total_seconds(), 1))
                try:
                    result = await asyncio.to_thread(run_weekly_crawl_once, ctx.config, ctx.client)
                    logger.info("weekly crawl forwarding succeeded: %s", result)
                except Exception:
                    # 실패해도 프로세스는 유지하고 다음 주기에 다시 시도한다.
                    logger.exception("weekly crawl forwarding failed")

        app.state.weekly_task = asyncio.create_task(_weekly_loop())
        try:
            yield
        finally:
            task = getattr(app.state, "weekly_task", None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="AI-Agent-Crawler Live Service", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        if request.url.path.startswith(API_V1_PREFIX):
            return v1_error(
                "COM_002",
                "요청 데이터 변환 과정에서 오류가 발생했습니다.",
                status_code=400,
            )
        return await request_validation_exception_handler(request, exc)

    app.include_router(create_legacy_router(ctx))
    app.include_router(create_v1_router(ctx))
    return app
