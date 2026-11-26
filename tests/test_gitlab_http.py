"""Task 1: GitLab retry policy, Retry-After parsing and typed request errors."""

from __future__ import annotations

import ssl
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from mira.exceptions import ProviderError
from mira.platforms.gitlab.http import (
    GitLabRequestError,
    GitLabRetryPolicy,
    parse_retry_after,
    retryable_exception,
    retryable_status,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_SAFE_METHODS = ("GET", "HEAD")
_WRITE_METHODS = ("POST", "PUT", "PATCH", "DELETE")

# Every status the spec names or implies: GET/HEAD retry only the transient
# set, writes retry only an explicit 429, everything else is terminal.
_STATUS_MATRIX = [
    (200, False),
    (204, False),
    (301, False),
    (302, False),
    (400, False),
    (401, False),
    (403, False),
    (404, False),
    (408, True),
    (409, False),
    (422, False),
    (429, True),
    (500, True),
    (501, False),
    (502, True),
    (503, True),
    (504, True),
    (505, False),
]


@pytest.mark.parametrize(
    ("method", "status", "expected"),
    [(method, status, expected) for method in _SAFE_METHODS for status, expected in _STATUS_MATRIX]
    + [
        ("get", 429, True),
        ("head", 503, True),
        ("Get", 500, True),
    ],
)
def test_retryable_status(method: str, status: int, expected: bool) -> None:
    assert retryable_status(method, status) is expected


@pytest.mark.parametrize("method", _WRITE_METHODS)
@pytest.mark.parametrize("status", [status for status, _ in _STATUS_MATRIX])
def test_retryable_status_writes_only_429(method: str, status: int) -> None:
    assert retryable_status(method, status) is (status == 429)


@pytest.mark.parametrize(
    ("method", "status", "expected"),
    [
        ("post", 429, True),
        ("Post", 503, False),
        ("delete", 408, False),
    ],
)
def test_retryable_status_write_methods_normalize(method: str, status: int, expected: bool) -> None:
    assert retryable_status(method, status) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("120", 120.0),
        ("0.5", 0.5),
        ("0", 0.0),
        (format_datetime(NOW + timedelta(seconds=120), usegmt=True), 120.0),
        (format_datetime(NOW - timedelta(seconds=120), usegmt=True), 0.0),
        # Valid HTTP-date without a timezone is normalized to UTC.
        ("Thu, 01 Jan 2026 12:02:00", 120.0),
        ("Thu, 01 Jan 2026 12:02:00 -0000", 120.0),
        ("-1", None),
        ("nan", None),
        ("NaN", None),
        ("inf", None),
        ("Infinity", None),
        ("not-a-date", None),
        ("1 2", None),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    result = parse_retry_after(value, now=NOW)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


_TRANSIENT_EXCEPTIONS = [
    pytest.param(lambda: httpx.ConnectError("connect failed"), id="ConnectError"),
    pytest.param(lambda: httpx.ConnectTimeout("connect timeout"), id="ConnectTimeout"),
    pytest.param(lambda: httpx.ReadError("read failed"), id="ReadError"),
    pytest.param(lambda: httpx.ReadTimeout("read timeout"), id="ReadTimeout"),
    pytest.param(lambda: httpx.WriteError("write failed"), id="WriteError"),
    pytest.param(lambda: httpx.WriteTimeout("write timeout"), id="WriteTimeout"),
    pytest.param(
        lambda: httpx.RemoteProtocolError("remote protocol error"), id="RemoteProtocolError"
    ),
]


@pytest.mark.parametrize("make_exc", _TRANSIENT_EXCEPTIONS)
def test_retryable_exception_safe_method(make_exc) -> None:
    assert retryable_exception("GET", make_exc()) is True


@pytest.mark.parametrize("make_exc", _TRANSIENT_EXCEPTIONS)
def test_retryable_exception_write_method_is_false(make_exc) -> None:
    assert retryable_exception("POST", make_exc()) is False


@pytest.mark.parametrize("method", ["GET", "HEAD", "POST"])
def test_retryable_exception_pool_timeout_is_false(method: str) -> None:
    assert retryable_exception(method, httpx.PoolTimeout("pool exhausted")) is False


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_retryable_exception_tls_verification_failure_is_false(method: str) -> None:
    cert_error = ssl.SSLCertVerificationError("certificate verify failed")
    exc = httpx.ConnectError("certificate verify failed")
    exc.__cause__ = cert_error
    assert retryable_exception(method, exc) is False


def test_retryable_exception_non_transported_errors_are_false() -> None:
    assert retryable_exception("GET", httpx.ProxyError("proxy denied")) is False
    assert retryable_exception("GET", httpx.UnsupportedProtocol("bad scheme")) is False


def test_policy_defaults_and_frozen() -> None:
    policy = GitLabRetryPolicy()
    assert policy.max_attempts == 5
    assert policy.deadline_seconds == 300.0
    assert policy.backoff_cap_seconds == 60.0
    with pytest.raises(FrozenInstanceError):
        policy.max_attempts = 2  # type: ignore[misc]


def test_request_error_carries_provider_fields() -> None:
    err = GitLabRequestError("exhausted retries", 3, 429)
    assert isinstance(err, ProviderError)
    assert err.reason == "exhausted retries"
    assert err.attempts == 3
    assert err.status_code == 429
    assert err.safe_message == str(err)
