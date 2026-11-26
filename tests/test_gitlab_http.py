"""Task 1: GitLab retry policy, Retry-After parsing and typed request errors."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import ssl
from collections.abc import AsyncIterator, Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from mira.exceptions import ProviderError
from mira.platforms.gitlab import http as gitlab_http
from mira.platforms.gitlab.auth import GitLabTokenAuth
from mira.platforms.gitlab.http import (
    GitLabHttpRuntime,
    GitLabRequestError,
    GitLabRetryPolicy,
    gitlab_http_scope,
    parse_retry_after,
    request_gitlab,
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


# ===========================================================================
# Task 2: pooled runtime, retry executor and shared throttle gates
# ===========================================================================

GITLAB_API = "https://gitlab.example.com/api/v4"


class FakeClock:
    """Deterministic clock/sleep pair injected through module helpers."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ManualClock:
    """Clock whose sleeps block until the test releases them."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self._events: list[asyncio.Event] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        event = asyncio.Event()
        self._events.append(event)
        self.sleeps.append(seconds)
        await event.wait()
        self.now += seconds

    def release_all(self) -> None:
        for event in self._events:
            event.set()


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(gitlab_http, "_monotonic", clock.time)
    monkeypatch.setattr(gitlab_http, "_sleep", clock.sleep)
    return clock


@pytest.fixture
def manual_clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    clock = ManualClock()
    monkeypatch.setattr(gitlab_http, "_monotonic", clock.time)
    monkeypatch.setattr(gitlab_http, "_sleep", clock.sleep)
    return clock


async def wait_until(predicate: Callable[[], bool], *, what: str) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


async def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


async def _async_chunks() -> AsyncIterator[bytes]:
    yield b"chunk"


class NonSeekableStream:
    """A one-shot binary stream with no seek, so retries cannot replay it."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._done = False

    def read(self, size: int = -1) -> bytes:
        if self._done:
            return b""
        self._done = True
        return self._payload


class BlockedCloseTransport(httpx.MockTransport):
    """A real transport whose ``aclose`` blocks until the test releases it."""

    def __init__(
        self,
        handler,
        *,
        entered: asyncio.Event,
        resume: asyncio.Event,
        completed: asyncio.Event,
    ) -> None:
        super().__init__(handler)
        self._entered = entered
        self._resume = resume
        self._completed = completed

    async def aclose(self) -> None:
        self._entered.set()
        await self._resume.wait()
        self._completed.set()


def make_runtime(handler, *, policy: GitLabRetryPolicy | None = None):
    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        created.append(client)
        return client

    return GitLabHttpRuntime(policy=policy, client_factory=factory), created


def state_for_token(runtime: GitLabHttpRuntime, token: str = "token"):
    api = gitlab_http._parse_api_url(GITLAB_API)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return runtime._state_for(digest, api)


# -- MockTransport retry/body behaviour -------------------------------------


async def test_get_503_then_200_waits_exact_retry_after(fake_clock: FakeClock) -> None:
    sends: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sends.append(request.url.path)
        if len(sends) == 1:
            return httpx.Response(503, headers={"Retry-After": "1.5"})
        return httpx.Response(200, json={"ok": True})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/projects")
        assert response.status_code == 200
        assert len(sends) == 2
        assert fake_clock.sleeps == [1.5]
    finally:
        await runtime.aclose()


async def test_get_503_then_200_sends_twice_with_fallback_backoff(
    fake_clock: FakeClock,
) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        if sends == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/projects")
        assert response.status_code == 200
        assert sends == 2
        assert len(fake_clock.sleeps) == 1
        assert 0.5 <= fake_clock.sleeps[0] <= 1.0
    finally:
        await runtime.aclose()


async def test_retried_response_is_closed_before_sleep(
    fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sends = 0
    events: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        if sends == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    real_close = gitlab_http._close_for_retry
    real_sleep = gitlab_http._sleep

    async def recording_close(response: httpx.Response) -> None:
        events.append("close")
        await real_close(response)

    async def recording_sleep(seconds: float) -> None:
        events.append("sleep")
        await real_sleep(seconds)

    monkeypatch.setattr(gitlab_http, "_close_for_retry", recording_close)
    monkeypatch.setattr(gitlab_http, "_sleep", recording_sleep)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/projects")
        assert response.status_code == 200
        assert events == ["close", "sleep"]
    finally:
        await runtime.aclose()


async def test_get_failures_exhaust_at_five_sends_and_four_sleeps(
    fake_clock: FakeClock,
) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(503)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/projects")
        assert excinfo.value.attempts == 5
        assert excinfo.value.status_code == 503
        assert sends == 5
        assert len(fake_clock.sleeps) == 4
        bounds = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
        for delay, (low, high) in zip(fake_clock.sleeps, bounds, strict=True):
            assert low <= delay <= high
    finally:
        await runtime.aclose()


async def test_get_transient_transport_failures_exhaust(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        raise httpx.ReadTimeout("read timed out")

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/projects")
        assert excinfo.value.attempts == 5
        assert sends == 5
        assert len(fake_clock.sleeps) == 4
    finally:
        await runtime.aclose()


async def test_429_then_200_waits_exactly_the_valid_header(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        if sends == 1:
            return httpx.Response(429, headers={"Retry-After": "0.75"})
        return httpx.Response(200, json={"ok": True})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200
        assert sends == 2
        assert fake_clock.sleeps == [0.75]
    finally:
        await runtime.aclose()


async def test_404_returns_immediately_without_retry(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(404, json={"message": "404 Not Found"})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/missing")
        assert response.status_code == 404
        assert sends == 1
        assert fake_clock.sleeps == []
    finally:
        await runtime.aclose()


async def test_post_503_sends_once_and_is_returned(fake_clock: FakeClock) -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        return httpx.Response(503, headers={"Retry-After": "5"})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token", GITLAB_API, "POST", f"{GITLAB_API}/notes", json={"body": "hello"}
        )
        assert response.status_code == 503
        assert len(bodies) == 1
        assert fake_clock.sleeps == []
    finally:
        await runtime.aclose()


async def test_post_read_timeout_sends_once_and_raises(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        raise httpx.ReadTimeout("write timed out")

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request(
                "token", GITLAB_API, "POST", f"{GITLAB_API}/notes", data={"body": "hi"}
            )
        assert excinfo.value.attempts == 1
        assert sends == 1
        assert fake_clock.sleeps == []
    finally:
        await runtime.aclose()


async def test_pool_timeout_is_terminal_without_retry(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        raise httpx.PoolTimeout("pool exhausted")

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 1
        assert sends == 1
        assert fake_clock.sleeps == []
    finally:
        await runtime.aclose()


async def test_tls_verification_failure_is_terminal_without_retry(
    fake_clock: FakeClock,
) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        cert_error = ssl.SSLCertVerificationError("certificate verify failed")
        exc = httpx.ConnectError("certificate verify failed")
        exc.__cause__ = cert_error
        raise exc

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 1
        assert sends == 1
        assert fake_clock.sleeps == []
    finally:
        await runtime.aclose()


async def test_post_json_429_replays_identical_body(fake_clock: FakeClock) -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        if len(bodies) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token",
            GITLAB_API,
            "POST",
            f"{GITLAB_API}/notes",
            json={"body": "hello world", "n": 1},
        )
        assert response.status_code == 200
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
        assert b"hello world" in bodies[0]
    finally:
        await runtime.aclose()


async def test_post_form_429_replays_identical_body(fake_clock: FakeClock) -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        if len(bodies) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token",
            GITLAB_API,
            "POST",
            f"{GITLAB_API}/notes",
            data={"body": "hello world", "n": "1"},
        )
        assert response.status_code == 200
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
        assert b"hello+world" in bodies[0]
    finally:
        await runtime.aclose()


@pytest.mark.parametrize(
    "stream_factory",
    [
        pytest.param(lambda: iter([b"chunk"]), id="sync-iterator"),
        pytest.param(_async_chunks, id="async-iterator"),
    ],
)
async def test_streaming_bodies_are_rejected(stream_factory) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError, match="non-replayable"):
            await runtime.request(
                "token",
                GITLAB_API,
                "POST",
                f"{GITLAB_API}/notes",
                content=stream_factory(),
            )
        assert sends == 0
    finally:
        await runtime.aclose()


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {"attachment": ("blob.bin", NonSeekableStream(b"payload"))},
            id="dict-tuple",
        ),
        pytest.param(
            [("attachment", ("blob.bin", NonSeekableStream(b"payload")))],
            id="list-pairs",
        ),
        pytest.param({"attachment": NonSeekableStream(b"payload")}, id="bare"),
    ],
)
async def test_non_seekable_nested_multipart_file_is_rejected(files) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError, match="non-replayable"):
            await runtime.request(
                "token",
                GITLAB_API,
                "POST",
                f"{GITLAB_API}/uploads",
                files=files,
            )
        assert sends == 0
    finally:
        await runtime.aclose()


async def test_seekable_nested_multipart_file_is_replayed(
    fake_clock: FakeClock,
) -> None:
    payload = io.BytesIO(b"file-bytes")
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        if len(bodies) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token",
            GITLAB_API,
            "POST",
            f"{GITLAB_API}/uploads",
            files={"attachment": ("blob.bin", payload)},
        )
        assert response.status_code == 200
        assert len(bodies) == 2
        assert b"file-bytes" in bodies[0]
        assert b"file-bytes" in bodies[1]
        assert b'filename="blob.bin"' in bodies[0]
        assert b'filename="blob.bin"' in bodies[1]
    finally:
        await runtime.aclose()


async def test_120_second_retry_after_is_not_truncated(fake_clock: FakeClock) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        if sends == 1:
            return httpx.Response(429, headers={"Retry-After": "120"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/archive")
        assert response.status_code == 200
        assert fake_clock.sleeps == [120.0]
    finally:
        await runtime.aclose()


async def test_read_timeout_is_applied_to_each_attempt(fake_clock: FakeClock) -> None:
    timeouts: list[dict[str, float]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(dict(request.extensions["timeout"]))
        if len(timeouts) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token", GITLAB_API, "GET", f"{GITLAB_API}/archive", read_timeout=120.0
        )
        assert response.status_code == 200
        assert [timeout["read"] for timeout in timeouts] == [120.0, 120.0]
        assert timeouts[0]["connect"] == 10.0
        assert timeouts[0]["write"] == 30.0
        assert timeouts[0]["pool"] == 5.0
    finally:
        await runtime.aclose()


async def test_server_directed_delay_beyond_deadline_records_cooldown_then_fails(
    fake_clock: FakeClock,
) -> None:
    start = fake_clock.now
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(429, headers={"Retry-After": "301"})

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 1
        assert excinfo.value.status_code == 429
        assert "deadline" in excinfo.value.reason
        assert sends == 1
        assert fake_clock.sleeps == []
        state = next(iter(runtime._states.values()))
        assert state.cooldown_not_before == pytest.approx(start + 301)
    finally:
        await runtime.aclose()


# -- deterministic shared-gate behaviour -----------------------------------


async def test_sibling_jobs_share_a_429_gate(manual_clock: ManualClock) -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        task = asyncio.current_task()
        calls.append(task.get_name() if task else "?")
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        first = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/a"), name="first"
        )
        await wait_until(lambda: len(manual_clock.sleeps) == 1, what="first retry sleep")
        assert calls == ["first"]

        second = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/b"), name="second"
        )
        await wait_until(lambda: len(manual_clock.sleeps) == 2, what="sibling gate wait")
        assert calls == ["first"]
        assert manual_clock.sleeps == [5.0, 5.0]

        manual_clock.release_all()
        responses = await asyncio.gather(first, second)
        assert [response.status_code for response in responses] == [200, 200]
        assert len(calls) == 3
    finally:
        await runtime.aclose()


async def test_cooldown_extensions_take_the_maximum(manual_clock: ManualClock) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "10"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        worker = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user"), name="worker"
        )
        await wait_until(lambda: len(manual_clock.sleeps) == 1, what="first retry sleep")
        state = next(iter(runtime._states.values()))
        assert state.cooldown_not_before == manual_clock.now + 10

        # A sibling's longer 429 must extend the gate; a shorter one must not
        # shorten it.
        gitlab_http._extend_cooldown(state, manual_clock.now + 20)
        assert state.cooldown_not_before == manual_clock.now + 20
        gitlab_http._extend_cooldown(state, manual_clock.now + 5)
        assert state.cooldown_not_before == manual_clock.now + 20

        manual_clock.release_all()
        await wait_until(lambda: len(manual_clock.sleeps) == 2, what="extended gate wait")
        assert manual_clock.sleeps == [10.0, 10.0]
        assert calls == 1

        manual_clock.release_all()
        response = await worker
        assert response.status_code == 200
        assert calls == 2
    finally:
        await runtime.aclose()


async def test_final_attempt_429_still_gates_siblings(fake_clock: FakeClock) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 5:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError) as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 5
        assert excinfo.value.status_code == 429
        assert calls == 5
        assert fake_clock.sleeps == [5.0, 5.0, 5.0, 5.0]
        state = next(iter(runtime._states.values()))
        assert state.cooldown_not_before == pytest.approx(fake_clock.now + 5)

        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/other")
        assert response.status_code == 200
        assert len(fake_clock.sleeps) == 5
        assert fake_clock.sleeps[-1] == 5.0
    finally:
        await runtime.aclose()


async def test_503_retry_after_gates_siblings_even_for_post(fake_clock: FakeClock) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(503, headers={"Retry-After": "5"})
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token", GITLAB_API, "POST", f"{GITLAB_API}/notes", json={"x": 1}
        )
        assert response.status_code == 503
        assert fake_clock.sleeps == []

        sibling = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert sibling.status_code == 200
        assert fake_clock.sleeps == [5.0]
    finally:
        await runtime.aclose()


async def test_retry_sleeps_do_not_hold_a_request_permit(
    fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    runtime, _ = make_runtime(handler)
    permit_values: list[int] = []
    original_sleep = gitlab_http._sleep

    async def observing_sleep(seconds: float) -> None:
        for state in runtime._states.values():
            permit_values.append(state.semaphore._value)
        await original_sleep(seconds)

    monkeypatch.setattr(gitlab_http, "_sleep", observing_sleep)
    try:
        with pytest.raises(GitLabRequestError):
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert permit_values == [gitlab_http._MAX_ACTIVE_REQUESTS] * 4
    finally:
        await runtime.aclose()


async def test_permit_is_rechecked_against_extended_gate(manual_clock: ManualClock) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    state = state_for_token(runtime)
    inner = state.semaphore
    release = asyncio.Event()

    async def hold_permit() -> None:
        await inner.acquire()
        try:
            await release.wait()
        finally:
            inner.release()

    holders = [
        asyncio.create_task(hold_permit(), name=f"holder-{index}")
        for index in range(gitlab_http._MAX_ACTIVE_REQUESTS)
    ]
    try:
        await wait_until(lambda: inner._value == 0, what="all request permits held")

        events: list[str] = []

        class RecordingSemaphore:
            async def acquire(self) -> None:
                await inner.acquire()
                events.append("acquire")

            def release(self) -> None:
                inner.release()
                events.append("release")

            @property
            def _value(self) -> int:
                return inner._value

        state.semaphore = RecordingSemaphore()

        worker = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user"), name="worker"
        )
        await wait_until(
            lambda: inner._waiters is not None and len(inner._waiters) > 0,
            what="worker queued for permit",
        )
        gitlab_http._extend_cooldown(state, manual_clock.now + 5)
        release.set()

        await wait_until(
            lambda: events == ["acquire", "release"], what="permit released after recheck"
        )
        await wait_until(lambda: manual_clock.sleeps == [5.0], what="gate wait after recheck")
        assert worker.done() is False

        manual_clock.release_all()
        await asyncio.sleep(0)
        response = await worker
        assert response.status_code == 200
    finally:
        release.set()
        await asyncio.gather(*holders, return_exceptions=True)
        await runtime.aclose()


# -- ownership and lifecycle -----------------------------------------------


async def test_concurrent_first_requests_create_one_client_per_key() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200)

    runtime, created = make_runtime(handler)
    try:
        tasks = [
            asyncio.create_task(
                runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/{index}")
            )
            for index in range(8)
        ]
        await started.wait()
        await wait_until(lambda: len(created) == 1, what="single pooled client")
        release.set()
        responses = await asyncio.gather(*tasks)
        assert all(response.status_code == 200 for response in responses)
        assert len(created) == 1
    finally:
        release.set()
        await runtime.aclose()


async def test_normalized_api_urls_share_pooled_state() -> None:
    runtime, created = make_runtime(ok_handler)
    variants = [
        "https://gitlab.example.com/api/v4",
        "https://GITLAB.EXAMPLE.COM:443/api/v4/",
        "https://gitlab.example.com/api/v4/",
    ]
    try:
        for api_url in variants:
            response = await runtime.request("token", api_url, "GET", f"{api_url}/user")
            assert response.status_code == 200
        assert len(created) == 1
        assert len(runtime._states) == 1
    finally:
        await runtime.aclose()


async def test_different_tokens_and_hosts_do_not_share_state() -> None:
    runtime, created = make_runtime(ok_handler)
    try:
        await runtime.request("token-a", GITLAB_API, "GET", f"{GITLAB_API}/a")
        await runtime.request("token-b", GITLAB_API, "GET", f"{GITLAB_API}/b")
        await runtime.request(
            "token-a",
            "https://other.example.com/api/v4",
            "GET",
            "https://other.example.com/api/v4/a",
        )
        assert len(created) == 3
        assert len(runtime._states) == 3
    finally:
        await runtime.aclose()


async def test_borrowed_scope_does_not_close_clients() -> None:
    runtime, created = make_runtime(ok_handler)
    async with gitlab_http_scope(runtime) as bound:
        assert bound is runtime
        response = await request_gitlab("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200
    assert created[0].is_closed is False
    await runtime.aclose()
    assert created[0].is_closed is True


async def test_owned_scope_closes_client_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(ok_handler), follow_redirects=False
        )
        created.append(client)
        return client

    monkeypatch.setattr(gitlab_http, "_default_client_factory", factory)
    holder: dict[str, GitLabHttpRuntime] = {}
    with pytest.raises(ValueError, match="boom"):
        async with gitlab_http_scope() as runtime:
            holder["runtime"] = runtime
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
            raise ValueError("boom")
    assert created[0].is_closed is True
    await holder["runtime"].aclose()


async def test_owned_scope_closes_client_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(ok_handler), follow_redirects=False
        )
        created.append(client)
        return client

    monkeypatch.setattr(gitlab_http, "_default_client_factory", factory)

    async def hold() -> None:
        async with gitlab_http_scope() as runtime:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    try:
        await wait_until(lambda: len(created) == 1, what="owned client creation")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert created[0].is_closed is True
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_use_after_close_fails_explicitly() -> None:
    runtime, _ = make_runtime(ok_handler)
    await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
    await runtime.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")


async def test_aclose_is_idempotent() -> None:
    runtime, created = make_runtime(ok_handler)
    await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
    await runtime.aclose()
    await runtime.aclose()
    assert created[0].is_closed is True


async def test_aclose_resumes_after_cancellation_with_multiple_clients() -> None:
    runtime, created = make_runtime(ok_handler)
    await runtime.request("token-a", GITLAB_API, "GET", f"{GITLAB_API}/a")
    await runtime.request(
        "token-b",
        "https://other.example.com/api/v4",
        "GET",
        "https://other.example.com/api/v4/b",
    )
    assert len(created) == 2

    entered = asyncio.Event()
    resume = asyncio.Event()
    original_close = created[0].aclose

    async def slow_close() -> None:
        entered.set()
        await resume.wait()
        await original_close()

    created[0].aclose = slow_close  # type: ignore[method-assign]
    try:
        closing = asyncio.create_task(runtime.aclose(), name="closing")
        await entered.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert created[0].is_closed is False
        assert created[1].is_closed is False

        resume.set()
        await wait_until(lambda: created[0].is_closed, what="shielded client close")
        await runtime.aclose()
        assert created[0].is_closed is True
        assert created[1].is_closed is True
    finally:
        resume.set()
        await runtime.aclose()


async def test_aclose_retry_awaits_same_transport_cleanup() -> None:
    entered = asyncio.Event()
    resume = asyncio.Event()
    completed = asyncio.Event()

    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        if not created:
            transport = BlockedCloseTransport(
                ok_handler, entered=entered, resume=resume, completed=completed
            )
        else:
            transport = httpx.MockTransport(ok_handler)
        client = httpx.AsyncClient(transport=transport, follow_redirects=False)
        created.append(client)
        return client

    runtime = GitLabHttpRuntime(client_factory=factory)
    try:
        await runtime.request("token-a", GITLAB_API, "GET", f"{GITLAB_API}/a")
        await runtime.request(
            "token-b",
            "https://other.example.com/api/v4",
            "GET",
            "https://other.example.com/api/v4/b",
        )
        assert len(created) == 2

        closing = asyncio.create_task(runtime.aclose(), name="first-close")
        await entered.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        # Retry immediately, without releasing the blocked transport cleanup.
        # It must await the retained cleanup task instead of starting a second
        # client.aclose() that returns because HTTPX already marked the client
        # closed.
        retry = asyncio.create_task(runtime.aclose(), name="retry-close")
        for _ in range(50):
            await asyncio.sleep(0)
            if created[1].is_closed:
                break
        assert retry.done() is False
        assert completed.is_set() is False
        assert created[1].is_closed is False

        resume.set()
        await wait_until(lambda: completed.is_set(), what="blocked transport cleanup")
        await retry
        assert created[0].is_closed is True
        assert created[1].is_closed is True
    finally:
        resume.set()
        await runtime.aclose()


def test_cross_event_loop_access_fails_explicitly() -> None:
    runtime, _ = make_runtime(ok_handler)

    async def first_loop() -> None:
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200
        await runtime.aclose()

    asyncio.run(first_loop())

    async def second_loop() -> None:
        await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")

    with pytest.raises(RuntimeError, match="event loop"):
        asyncio.run(second_loop())


def test_request_gitlab_without_scope_uses_and_closes_temporary_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(ok_handler), follow_redirects=False
        )
        created.append(client)
        return client

    monkeypatch.setattr(gitlab_http, "_default_client_factory", factory)

    async def main() -> None:
        response = await request_gitlab("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200

    asyncio.run(main())
    assert len(created) == 1
    assert created[0].is_closed is True


async def test_request_gitlab_uses_bound_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_factory() -> httpx.AsyncClient:
        raise AssertionError("a bound scope must not create a temporary runtime")

    monkeypatch.setattr(gitlab_http, "_default_client_factory", forbidden_factory)
    runtime, created = make_runtime(ok_handler)
    try:
        async with gitlab_http_scope(runtime):
            response = await request_gitlab("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200
        assert len(created) == 1
    finally:
        await runtime.aclose()


# -- deadline and cancellation safety --------------------------------------


async def test_semaphore_queue_respects_overall_deadline() -> None:
    runtime, _ = make_runtime(ok_handler, policy=GitLabRetryPolicy(deadline_seconds=0.05))
    state = state_for_token(runtime)
    acquired: list[bool] = []

    async def hold_permit() -> None:
        await state.semaphore.acquire()
        try:
            acquired.append(True)
            await asyncio.sleep(3600)
        finally:
            state.semaphore.release()

    holders = [
        asyncio.create_task(hold_permit(), name=f"holder-{index}")
        for index in range(gitlab_http._MAX_ACTIVE_REQUESTS)
    ]
    try:
        await wait_until(
            lambda: len(acquired) == gitlab_http._MAX_ACTIVE_REQUESTS,
            what="all permits held",
        )
        started = asyncio.get_running_loop().time()
        with pytest.raises(GitLabRequestError, match="deadline"):
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert asyncio.get_running_loop().time() - started < 5
        assert state.semaphore._value == 0
    finally:
        for holder in holders:
            holder.cancel()
        await asyncio.gather(*holders, return_exceptions=True)
        await runtime.aclose()


async def test_network_wait_respects_overall_deadline() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200)

    runtime, _ = make_runtime(handler, policy=GitLabRetryPolicy(deadline_seconds=0.05))
    try:
        started = asyncio.get_running_loop().time()
        with pytest.raises(GitLabRequestError, match="deadline") as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 1
        assert asyncio.get_running_loop().time() - started < 5
    finally:
        await runtime.aclose()


async def test_cooldown_delay_beyond_deadline_records_before_failing(
    fake_clock: FakeClock,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "10"})

    runtime, _ = make_runtime(handler, policy=GitLabRetryPolicy(deadline_seconds=0.05))
    try:
        with pytest.raises(GitLabRequestError, match="deadline") as excinfo:
            await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert excinfo.value.attempts == 1
        assert fake_clock.sleeps == []
        state = next(iter(runtime._states.values()))
        assert state.cooldown_not_before == pytest.approx(fake_clock.now + 10)
    finally:
        await runtime.aclose()


async def test_cancellation_releases_permit_and_keeps_shared_state() -> None:
    entered = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release_handler.wait()
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    state = state_for_token(runtime)
    gitlab_http._extend_cooldown(state, asyncio.get_running_loop().time() - 1.0)
    gate_before = state.cooldown_not_before
    try:
        task = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.semaphore._value == gitlab_http._MAX_ACTIVE_REQUESTS
        assert state.cooldown_not_before == gate_before

        release_handler.set()
        response = await runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert response.status_code == 200
    finally:
        release_handler.set()
        await runtime.aclose()


async def test_cancellation_during_retry_sleep_keeps_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "5"})

    runtime, _ = make_runtime(handler)
    real_sleep = gitlab_http._sleep
    slept = asyncio.Event()

    async def interruptible_sleep(seconds: float) -> None:
        slept.set()
        await real_sleep(seconds)

    monkeypatch.setattr(gitlab_http, "_sleep", interruptible_sleep)
    try:
        task = asyncio.create_task(
            runtime.request("token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        )
        await slept.wait()
        state = next(iter(runtime._states.values()))
        gate = state.cooldown_not_before
        assert gate > asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.cooldown_not_before == gate
        assert state.semaphore._value == gitlab_http._MAX_ACTIVE_REQUESTS
    finally:
        await runtime.aclose()


# -- target validation and credential safety --------------------------------


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("https://evil.example.com/api/v4/user", "origin"),
        ("http://gitlab.example.com/api/v4/user", "origin"),
        ("https://gitlab.example.com:8443/api/v4/user", "origin"),
        ("https://gitlab.example.com/other/user", "root"),
        ("https://gitlab.example.com/api/v4other/user", "root"),
        ("https://user:pass@gitlab.example.com/api/v4/user", "invalid"),
    ],
)
async def test_targets_outside_configured_origin_or_root_are_rejected(
    target: str, match: str
) -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError, match=match):
            await runtime.request("secret-token", GITLAB_API, "GET", target)
        assert sends == 0
    finally:
        await runtime.aclose()


async def test_relative_target_is_resolved_under_api_root() -> None:
    seen: str | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = request.url.path
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request("token", GITLAB_API, "GET", "projects/1")
        assert response.status_code == 200
        assert seen == "/api/v4/projects/1"
    finally:
        await runtime.aclose()


async def test_absolute_dot_segment_target_cannot_escape_api_root() -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError, match="root"):
            await runtime.request(
                "secret-token",
                GITLAB_API,
                "GET",
                "https://gitlab.example.com/api/v4/../../admin",
            )
        assert sends == 0
    finally:
        await runtime.aclose()


async def test_scheme_relative_dot_segment_target_cannot_escape_api_root() -> None:
    sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sends
        sends += 1
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        with pytest.raises(GitLabRequestError, match="root"):
            await runtime.request(
                "secret-token",
                GITLAB_API,
                "GET",
                "//gitlab.example.com/api/v4/../user",
            )
        assert sends == 0
    finally:
        await runtime.aclose()


async def test_dot_segments_are_canonicalized_before_dispatch() -> None:
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200)

    runtime, _ = make_runtime(handler)
    try:
        response = await runtime.request(
            "token",
            GITLAB_API,
            "GET",
            "https://gitlab.example.com/api/v4/projects/../user",
        )
        assert response.status_code == 200
        assert paths == ["/api/v4/user"]
    finally:
        await runtime.aclose()


async def test_redirects_are_never_followed_with_credentials() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/v4/start":
            return httpx.Response(302, headers={"Location": f"{GITLAB_API}/next"})
        return httpx.Response(200)

    created: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        created.append(client)
        return client

    runtime = GitLabHttpRuntime(client_factory=factory)
    try:
        response = await runtime.request("secret", GITLAB_API, "GET", f"{GITLAB_API}/start")
        assert response.status_code == 302
        assert calls == ["/api/v4/start"]
        assert len(created) == 1
    finally:
        await runtime.aclose()


async def test_logs_and_errors_exclude_token_query_and_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    runtime, _ = make_runtime(handler)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger="mira.platforms.gitlab.http"),
            pytest.raises(GitLabRequestError),
        ):
            await runtime.request(
                "super-secret-token",
                GITLAB_API,
                "GET",
                f"{GITLAB_API}/notes?private_token=super-secret-token&q=1",
                json={"body": "super-secret-token"},
            )
        text = caplog.text
        assert "super-secret-token" not in text
        assert "private_token" not in text
        assert "q=1" not in text
    finally:
        await runtime.aclose()


async def test_state_keys_use_non_reversible_token_digest() -> None:
    runtime, _ = make_runtime(ok_handler)
    try:
        await runtime.request("super-secret-token", GITLAB_API, "GET", f"{GITLAB_API}/user")
        assert runtime._states
        for key in runtime._states:
            assert "super-secret-token" not in repr(key)
    finally:
        await runtime.aclose()


# -- Task 1 review carry-over: HEAD/lowercase and nested TLS causes ---------


@pytest.mark.parametrize("method", ["get", "Get", "head", "Head", "HEAD"])
def test_retryable_exception_safe_method_case_insensitive(method: str) -> None:
    assert retryable_exception(method, httpx.ReadTimeout("timeout")) is True


@pytest.mark.parametrize("method", ["post", "Post", "put", "delete"])
def test_retryable_exception_write_method_case_insensitive(method: str) -> None:
    assert retryable_exception(method, httpx.ConnectError("connect failed")) is False


def test_tls_failure_through_nested_context_chain_is_false() -> None:
    cert_error = ssl.SSLCertVerificationError("certificate verify failed")
    inner = httpx.ConnectError("wrapped failure")
    inner.__context__ = cert_error
    outer = httpx.ConnectError("outer failure")
    outer.__cause__ = inner
    assert retryable_exception("GET", outer) is False


def test_tls_failure_as_context_only_is_false() -> None:
    cert_error = ssl.SSLCertVerificationError("certificate verify failed")
    wrapper = httpx.ConnectError("wrapper")
    wrapper.__context__ = cert_error
    assert retryable_exception("HEAD", wrapper) is False


# -- Task 3: identity consumer routing --------------------------------------


async def test_auth_identity_recovers_from_503_then_200() -> None:
    sends: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sends.append(request)
        if len(sends) == 1:
            return httpx.Response(503, headers={"retry-after": "0"})
        return httpx.Response(200, json={"username": "project_9_bot"})

    runtime, _ = make_runtime(handler)
    try:
        async with gitlab_http_scope(runtime):
            auth = GitLabTokenAuth("token", GITLAB_API)
            assert await auth.get_bot_identity() == "project_9_bot"
            # The identity is cached; no second request.
            assert await auth.get_bot_identity() == "project_9_bot"
    finally:
        await runtime.aclose()

    assert len(sends) == 2
    assert sends[0].url == sends[1].url
    assert sends[0].headers["private-token"] == "token"


async def test_auth_identity_exhaustion_returns_none_without_partial_truth(
    fake_clock: FakeClock,
) -> None:
    sends: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sends.append(request)
        return httpx.Response(503, headers={"retry-after": "0"})

    runtime, _ = make_runtime(handler)
    try:
        async with gitlab_http_scope(runtime):
            auth = GitLabTokenAuth("token", GITLAB_API)
            assert await auth.get_bot_identity() is None
    finally:
        await runtime.aclose()

    assert len(sends) == 5
