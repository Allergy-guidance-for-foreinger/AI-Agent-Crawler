"""요청 language 정규화 단위 테스트."""

from __future__ import annotations

import pytest

from app.services.ops import normalize_request_language, validate_accept_language


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ko", "ko"),
        ("en", "en"),
        ("en-US", "en"),
        ("zh-CN", "zh-CN"),
        ("zh-cn", "zh-CN"),
        ("vi", "vi"),
        ("ja", "ja"),
        ("ja-JP", "ja"),
    ],
)
def test_normalize_request_language(raw: str, expected: str) -> None:
    assert normalize_request_language(raw) == expected


def test_normalize_request_language_rejects_empty() -> None:
    with pytest.raises(ValueError, match="language는 필수"):
        normalize_request_language("")


def test_normalize_request_language_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="지원하지 않는 language"):
        normalize_request_language("fr")


def test_validate_accept_language_delegates_to_normalize() -> None:
    validate_accept_language("en-US")
