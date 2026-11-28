"""Shared, provider-neutral HTTP execution and retry policy."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    DoAttempt,
    DoSleep,
    RetryCallState,
    RetryError,
    retry_if_exception,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)

from mira.exceptions import MiraError

logger = logging.getLogger(__name__)

_SAFE_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_READ_METHODS = frozenset({"GET", "HEAD"})
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 30.0
_DEFAULT_WRITE_TIMEOUT = 30.0
_DEFAULT_POOL_TIMEOUT = 5.0


class HttpRequestError(MiraError):
    """An HTTP request could not complete within its retry budget."""

    def __init__(
        self,
        reason: str,
        attempts: int,
        status_code: int | None = None,
        url: str | None = None,
    ) -> None:
        self.reason = reason
        self.attempts = attempts
        self.status_code = status_code
        self.url = url
        safe = f"{reason} (after {attempts} attempt{'s' if attempts != 1 else ''})"
        super().__init__(safe)


def parse_retry_after(value: str | None, *, now: datetime) -> float | None:
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return max(0.0, (target - now).total_seconds())
    if seconds < 0 or seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return seconds


@dataclass(frozen=True, slots=True)
class HttpRetryPolicy:
    """Bounded retry policy for one logical HTTP request."""

    max_attempts: int = 5
    deadline_seconds: float = 300.0
    backoff_cap_seconds: float = 60.0
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep)

    def delay_for(self, attempt: int) -> float:
        return min(self.backoff_cap_seconds, 2.0 ** (attempt - 1)) / 2.0


class RetryAfterWait(wait_fixed):
    """Tenacity wait strategy that prefers an HTTP ``Retry-After`` value."""

    def __init__(
        self,
        *,
        fallback: Callable[[int], float] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._fallback = fallback
        self._now = now or (lambda: datetime.now(UTC))
        super().__init__(0.0)

    def __call__(self, retry_state: RetryCallState) -> float:
        response: httpx.Response | None = None
        if retry_state.outcome is not None and not retry_state.outcome.failed:
            value = retry_state.outcome.result()
            if isinstance(value, httpx.Response):
                response = value
        if response is not None:
            delay = parse_retry_after(response.headers.get("retry-after"), now=self._now())
            if delay is not None:
                return delay
        return self._fallback(retry_state.attempt_number) if self._fallback else 0.0


def _is_retryable_transport_error(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TransportError)


def _retryable_response(method: str, response: httpx.Response) -> bool:
    if response.status_code == 429:
        return True
    return method.upper() in _READ_METHODS and response.status_code in _SAFE_RETRY_STATUS


def _cooldown_key(origin: str, credential: str) -> str:
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    return f"{origin}\n{digest}"


@dataclass
class _EndpointCooldown:
    not_before: float = 0.0


class HttpService:
    """Own pooled clients and coordinate throttles by origin and credential."""

    def __init__(
        self,
        *,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._sleep = sleep
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cooldowns: dict[str, _EndpointCooldown] = {}
        self._closed = False

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault(
            "limits",
            httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=30),
        )
        kwargs.setdefault(
            "timeout",
            httpx.Timeout(
                connect=_DEFAULT_CONNECT_TIMEOUT,
                read=_DEFAULT_READ_TIMEOUT,
                write=_DEFAULT_WRITE_TIMEOUT,
                pool=_DEFAULT_POOL_TIMEOUT,
            ),
        )
        kwargs.setdefault("follow_redirects", False)
        return httpx.AsyncClient(**kwargs)

    def _cooldown_key(self, origin: str, credential: str) -> str:
        return _cooldown_key(origin, credential)

    def _record_cooldown(self, key: str, seconds: float) -> None:
        state = self._cooldowns.setdefault(key, _EndpointCooldown())
        state.not_before = max(state.not_before, asyncio.get_running_loop().time() + seconds)

    async def _wait_for_cooldown(self, key: str, deadline: float) -> None:
        while True:
            state = self._cooldowns.get(key)
            if state is None:
                return
            remaining = state.not_before - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            if remaining > deadline - asyncio.get_running_loop().time():
                raise HttpRequestError("overall deadline exceeded before shared cooldown", 1)
            await self._sleep(remaining)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.aclose()

    async def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        credential: str = "",
        content: bytes | str | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        policy: HttpRetryPolicy,
        client_kwargs: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if self._closed:
            raise RuntimeError("HTTP service is closed")
        if policy.max_attempts < 1 or policy.deadline_seconds <= 0:
            raise HttpRequestError("retry policy allows no attempts", 0)

        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + policy.deadline_seconds
        split = urlsplit(url)
        origin = f"{split.scheme}://{split.netloc}"
        cooldown_key = self._cooldown_key(origin, credential)
        if cooldown_key not in self._locks:
            self._locks[cooldown_key] = asyncio.Lock()
        request_headers = dict(headers or {})

        async def send_once() -> httpx.Response:
            await self._wait_for_cooldown(cooldown_key, deadline)
            key = _cooldown_key(origin, credential)
            client = self._clients.get(key)
            if client is None:
                client = self._client_factory(**(client_kwargs or {}))
                self._clients[key] = client
            kwargs: dict[str, Any] = {"headers": request_headers}
            if content is not None:
                kwargs["content"] = content
            if json is not None:
                kwargs["json"] = json
            if params is not None:
                kwargs["params"] = params
            return await client.request(method, url, **kwargs)

        def should_retry(result: httpx.Response) -> bool:
            if not _retryable_response(method, result):
                return False
            delay = parse_retry_after(result.headers.get("retry-after"), now=datetime.now(UTC))
            if delay is None:
                delay = policy.delay_for(retry_state.attempt_number)
            self._record_cooldown(cooldown_key, delay)
            return True

        async def close_before_retry(result: httpx.Response) -> httpx.Response:
            await result.aread()
            await result.aclose()
            return result

        retrying = AsyncRetrying(
            stop=stop_after_attempt(policy.max_attempts),
            wait=RetryAfterWait(fallback=lambda attempt: policy.delay_for(attempt)),
            retry=(
                retry_if_exception(_is_retryable_transport_error) | retry_if_result(should_retry)
            ),
            reraise=False,
            sleep=policy.sleep,
        )

        attempts = 0
        retrying.begin()
        retry_state = RetryCallState(retry_object=retrying, fn=None, args=(), kwargs={})
        while True:
            try:
                action = await retrying.iter(retry_state=retry_state)
            except RetryError as exc:
                failed = exc.last_attempt.exception()
                status_code: int | None = None
                if isinstance(failed, httpx.HTTPError):
                    reason = f"terminal transport failure: {type(failed).__name__}"
                else:
                    reason = "exhausted retries"
                    outcome = retry_state.outcome
                    if outcome is not None and not outcome.failed:
                        value = outcome.result()
                        if isinstance(value, httpx.Response):
                            status_code = value.status_code
                raise HttpRequestError(
                    reason,
                    attempts or policy.max_attempts,
                    status_code=status_code,
                    url=url,
                ) from (failed if isinstance(failed, BaseException) else exc)
            if isinstance(action, DoAttempt):
                attempts = retry_state.attempt_number
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise HttpRequestError("overall deadline exceeded", attempts, url=url)
                try:
                    async with asyncio.timeout(remaining):
                        response = await send_once()
                except TimeoutError:
                    raise HttpRequestError("overall deadline exceeded", attempts, url=url) from None
                except httpx.TransportError as exc:
                    if _is_retryable_transport_error(exc) and method.upper() in _READ_METHODS:
                        retry_state.set_exception((type(exc), exc, exc.__traceback__))
                        continue
                    raise HttpRequestError(
                        f"terminal transport failure: {type(exc).__name__}", attempts, url=url
                    ) from exc
                except httpx.HTTPError as exc:
                    raise HttpRequestError(
                        f"terminal transport failure: {type(exc).__name__}", attempts, url=url
                    ) from exc
                if (
                    method.upper() not in _READ_METHODS
                    and response.status_code != 429
                    and response.status_code in _SAFE_RETRY_STATUS
                ):
                    await close_before_retry(response)
                    raise HttpRequestError(
                        f"terminal transport failure: HTTP {response.status_code}",
                        attempts,
                        status_code=response.status_code,
                        url=url,
                    )
                response = await close_before_retry(response)
                retry_state.set_result(response)
            elif isinstance(action, DoSleep):
                delay = float(action)
                if delay > deadline - loop.time():
                    status_code = None
                    outcome = retry_state.outcome
                    try:
                        status_code = outcome.result().status_code if outcome else None
                    except BaseException:
                        status_code = None
                    raise HttpRequestError(
                        "overall deadline exceeded before retry wait",
                        attempts,
                        status_code=status_code,
                        url=url,
                    )
                retry_state.prepare_for_next_attempt()
                await policy.sleep(delay)
            else:
                break

        outcome = retry_state.outcome
        status_code: int | None = None
        try:
            status_code = outcome.result().status_code if outcome and not outcome.failed else None
        except BaseException:
            status_code = None
        if outcome is None:
            raise HttpRequestError(
                "exhausted retries",
                attempts or policy.max_attempts,
                status_code=status_code,
                url=url,
            )
        if outcome.failed:
            exc = outcome.exception()
            raise HttpRequestError(
                f"terminal transport failure: {type(exc).__name__}",
                attempts or policy.max_attempts,
                url=url,
            ) from exc
        return outcome.result()
