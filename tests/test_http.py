from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from tenacity import RetryCallState

from mira.exceptions import MiraError
from mira.http import (
    HttpRequestError,
    HttpRetryPolicy,
    HttpService,
    RetryAfterWait,
    parse_retry_after,
)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_parse_retry_after_accepts_seconds_and_http_date() -> None:
    now = datetime(2026, 11, 28, tzinfo=UTC)
    assert parse_retry_after("1.5", now=now) == 1.5
    assert parse_retry_after("0", now=now) == 0
    future = (now + timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(future, now=now) == pytest.approx(45, abs=0.5)


@pytest.mark.parametrize(
    "value",
    ["-1", "NaN", "inf", "next year", ""],
)
async def test_parse_retry_after_rejects_invalid_values(value: str) -> None:
    now = datetime(2026, 11, 28, tzinfo=UTC)
    assert parse_retry_after(value, now=now) is None


async def test_retry_after_wait_uses_server_directed_delay() -> None:
    retry_after = RetryAfterWait(fallback=lambda attempt: 99.0)
    state = RetryCallState(None, None, args=(), kwargs={})
    response = httpx.Response(
        429,
        headers={"retry-after": "0.25"},
        request=httpx.Request("GET", "https://api.example.test"),
    )
    state.outcome = _Outcome(response)
    assert retry_after(state) == 0.25


class _Outcome:
    def __init__(self, value: object) -> None:
        self._value = value
        self.failed = False

    def result(self) -> object:
        return self._value

    def exception(self) -> BaseException | None:
        return None


async def test_service_retries_read_and_stops_at_attempt_limit() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(
        max_attempts=3, deadline_seconds=10, backoff_cap_seconds=0.01, sleep=asyncio.sleep
    )
    with pytest.raises(HttpRequestError) as exc_info:
        await service.request("https://api.example.test/v1/read", method="GET", policy=policy)
    assert attempts == 3
    assert exc_info.value.attempts == 3
    assert exc_info.value.status_code == 503


async def test_service_honours_retry_after_and_extends_shared_cooldown() -> None:
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(asyncio.get_running_loop().time())
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0.05"})
        return httpx.Response(200)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(
        max_attempts=3, deadline_seconds=10, backoff_cap_seconds=1, sleep=fake_sleep
    )
    response = await service.request(
        "https://api.example.test/v1/read", method="GET", policy=policy
    )
    assert response.status_code == 200
    assert sleep_calls == [pytest.approx(0.05)]
    assert len(calls) == 2


async def test_service_does_not_retry_before_server_wait() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"retry-after": "10"})

    calls: list[float] = []

    async def fail_fast_sleep(seconds: float) -> None:
        raise AssertionError(f"unexpected sleep: {seconds}")

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(
        max_attempts=2, deadline_seconds=0.2, backoff_cap_seconds=1, sleep=fail_fast_sleep
    )
    with pytest.raises(HttpRequestError):
        await service.request("https://api.example.test/v1/read", method="GET", policy=policy)
    assert calls == []


async def test_service_does_not_replay_write_after_transport_failure() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=3, deadline_seconds=10, backoff_cap_seconds=0.01)
    with pytest.raises(HttpRequestError):
        await service.request(
            "https://api.example.test/v1/write", method="POST", content=b"payload", policy=policy
        )
    assert attempts == 1


async def test_service_replays_write_after_explicit_429() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(204)

    seen_bodies: list[bytes] = []

    async def handler_capture(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(request.content)
        return handler(request)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=_transport(handler_capture), **kwargs
        )
    )
    policy = HttpRetryPolicy(max_attempts=3, deadline_seconds=10, backoff_cap_seconds=0.01)
    response = await service.request(
        "https://api.example.test/v1/write",
        method="POST",
        content=b"payload",
        credential="token",
        policy=policy,
    )
    assert response.status_code == 204
    assert seen_bodies == [b"payload", b"payload"]


async def test_service_does_not_replay_write_after_503() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, headers={"retry-after": "0"})

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=3, deadline_seconds=10, backoff_cap_seconds=0.01)
    with pytest.raises(HttpRequestError):
        await service.request(
            "https://api.example.test/v1/write",
            method="POST",
            content=b"payload",
            credential="token",
            policy=policy,
        )
    assert attempts == 1


async def test_service_isolates_cooldowns_by_origin_and_credential() -> None:
    service = HttpService()
    key_a = service._cooldown_key("https://one.test", "token-a")
    key_b = service._cooldown_key("https://two.test", "token-a")
    key_c = service._cooldown_key("https://one.test", "token-b")
    assert len({key_a, key_b, key_c}) == 3


async def test_service_cancellation_does_not_extend_shared_cooldown() -> None:
    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.sleep(10)
        return httpx.Response(200)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=10)
    task = asyncio.create_task(
        service.request("https://api.example.test/v1/read", policy=policy, credential="token")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        service._cooldowns.get(service._cooldown_key("https://api.example.test", "token")) is None
        or service._cooldowns[service._cooldown_key("https://api.example.test", "token")].not_before
        == 0
    )


async def test_service_close_is_idempotent() -> None:
    service = HttpService()
    await service.aclose()
    await service.aclose()


async def test_service_reuses_real_http_connections() -> None:
    connections = 0

    barrier = asyncio.Barrier(3)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
        body = b'{"ok":true}'
        while True:
            await barrier.wait()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break

    server = await asyncio.start_server(handler, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    service = HttpService()
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=5)
    responses = await asyncio.gather(
        *[
            service.request(f"http://127.0.0.1:{port}/read", policy=policy, credential="token")
            for _ in range(3)
        ]
    )
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert connections <= 3
    await service.aclose()
    server.close()
    await server.wait_closed()


async def test_request_error_is_provider_safe() -> None:
    error = HttpRequestError("terminal transport failure", 2)
    assert isinstance(error, MiraError)
    assert error.safe_message.startswith("terminal transport failure (after 2 attempts)")
