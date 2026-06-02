"""이미지 업로드 크기 제한 단위 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.api.routes.live import (
    _reject_oversized_upload_before_read,
    _validate_image_upload_v1,
)
from app.config.runtime import MAX_IMAGE_SIZE, MAX_IMAGE_SIZE_MB, NGINX_CLIENT_MAX_BODY_SIZE


def test_max_image_size_is_25mb() -> None:
    assert MAX_IMAGE_SIZE_MB == 25
    assert MAX_IMAGE_SIZE == 25 * 1024 * 1024


def test_validate_image_rejects_over_limit() -> None:
    over = b"x" * (MAX_IMAGE_SIZE + 1)
    ok, err = _validate_image_upload_v1(over, "image/jpeg")
    assert ok is False
    assert err is not None
    assert err.status_code == 413


def test_validate_image_accepts_at_limit() -> None:
    at_limit = b"x" * MAX_IMAGE_SIZE
    ok, err = _validate_image_upload_v1(at_limit, "image/jpeg")
    assert ok is True
    assert err is None


def test_nginx_body_size_has_headroom_over_app_limit() -> None:
    assert NGINX_CLIENT_MAX_BODY_SIZE == "30M"


def test_reject_oversized_upload_before_read() -> None:
    image = MagicMock()
    image.size = MAX_IMAGE_SIZE + 1
    resp = _reject_oversized_upload_before_read(image)
    assert resp is not None
    assert resp.status_code == 413


def test_reject_oversized_upload_allows_unknown_size() -> None:
    image = MagicMock()
    image.size = None
    assert _reject_oversized_upload_before_read(image) is None
