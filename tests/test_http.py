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
    http_service_scope,
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


async def test_service_records_final_attempt_cooldown_for_sibling_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0.05"})

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=1)
    with pytest.raises(HttpRequestError):
        await service.request("https://api.example.test/v1/read", policy=policy, credential="token")
    key = service._cooldown_key("https://api.example.test", "token")
    assert service._cooldowns[key].not_before > asyncio.get_running_loop().time()


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


async def test_service_applies_transport_retries_with_retry_after_wait() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return httpx.Response(200)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=2, deadline_seconds=5, sleep=fake_sleep)
    response = await service.request("https://api.example.test/v1/read", policy=policy)
    assert response.status_code == 200
    assert len(sleeps) == 1


async def test_service_uses_per_request_timeout_override() -> None:
    seen: list[float | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=5)
    await service.request("https://api.example.test/v1/read", policy=policy, timeout=1.5)
    assert seen[0]["read"] == 1.5


async def test_service_scope_binds_borrowed_runtime_without_closing_it() -> None:
    service = HttpService()
    seen: list[HttpService | None] = []

    async def use() -> None:
        async with http_service_scope(service) as bound:
            seen.append(bound)

    await use()
    assert seen == [service]
    assert not service._closed
    await service.aclose()


async def test_service_scope_owns_temporary_runtime_when_unbound() -> None:
    seen: list[HttpService | None] = []

    async def use() -> None:
        async with http_service_scope() as owned:
            seen.append(owned)

    await use()
    assert len(seen) == 1
    assert seen[0] is not None and seen[0]._closed


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


async def test_far_future_retry_after_is_clamped_to_request_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"retry-after": "1000000"})

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=0.5, sleep=asyncio.sleep)
    with pytest.raises(HttpRequestError):
        await service.request("https://api.example.test/v1/read", policy=policy, credential="token")
    key = service._cooldown_key("https://api.example.test", "token")
    remaining = service._cooldowns[key].not_before - asyncio.get_running_loop().time()
    assert remaining <= 0.5


async def test_client_configuration_does_not_leak_between_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/target"})
        return httpx.Response(200)

    service = HttpService(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=_transport(handler), **kwargs)
    )
    policy = HttpRetryPolicy(max_attempts=1, deadline_seconds=5)
    followed = await service.request(
        "https://api.example.test/redirect", policy=policy, follow_redirects=True
    )
    assert followed.status_code == 200
    # A default request must still use the non-redirecting client.
    plain = await service.request("https://api.example.test/redirect", policy=policy)
    assert plain.status_code == 302

    configured = await service.request(
        "https://api.example.test/redirect",
        policy=policy,
        client_kwargs={"follow_redirects": True},
    )
    assert configured.status_code == 200
    after = await service.request("https://api.example.test/redirect", policy=policy)
    assert after.status_code == 302


class _FlakyDrainResponse:
    def __init__(self, *, fail_drain: bool) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.fail_drain = fail_drain
        self.closed = False

    async def aread(self) -> None:
        if self.fail_drain:
            raise httpx.ReadTimeout("body read timed out")

    async def aclose(self) -> None:
        self.closed = True


class _FlakyDrainClient:
    def __init__(self) -> None:
        self.requests = 0
        self.responses: list[_FlakyDrainResponse] = []

    async def request(self, method: str, url: str, **kwargs) -> _FlakyDrainResponse:
        self.requests += 1
        response = _FlakyDrainResponse(fail_drain=self.requests == 1)
        self.responses.append(response)
        return response


async def test_response_drain_failure_is_retried_for_reads() -> None:
    client = _FlakyDrainClient()
    service = HttpService(client_factory=lambda **kwargs: client)
    policy = HttpRetryPolicy(max_attempts=2, deadline_seconds=5, backoff_cap_seconds=0.01)
    response = await service.request("https://api.example.test/v1/read", policy=policy)
    assert response.status_code == 200
    assert client.requests == 2
    assert client.responses[0].closed
