"""이미지 업로드 크기 제한 단위 테스트."""

from __future__ import annotations

from app.api.routes.live import _validate_image_upload_v1
from app.config.runtime import MAX_IMAGE_SIZE, MAX_IMAGE_SIZE_MB


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
