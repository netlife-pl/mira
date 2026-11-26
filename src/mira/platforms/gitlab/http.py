"""GitLab HTTP policy, pooled runtime and shared retry executor.

The pure helpers define retry decisions and server-directed delays;
``GitLabHttpRuntime`` consumes that policy for every logical request, pooling
clients and coordinating throttle cooldowns per normalized API URL and token.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import ssl
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from mira.exceptions import ProviderError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitLabRetryPolicy:
    """Bounded retry policy shared by every GitLab logical request."""

    max_attempts: int = 5
    deadline_seconds: float = 300.0
    backoff_cap_seconds: float = 60.0


# GET/HEAD blanket-retry statuses. Writes only ever retry an explicit 429;
# everything else (including 5xx and redirects) is terminal for them.
_SAFE_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_WRITE_RETRY_STATUS = 429

# Transient remote transport failures. PoolTimeout, TLS certificate
# verification failures, proxy errors and unsupported protocols are excluded.
_TRANSIENT_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
)


def retryable_status(method: str, status: int) -> bool:
    """Whether ``method`` may retry a response with ``status``.

    Method comparison is case-insensitive; GET/HEAD get the transient status
    matrix, every other method retries only 429.
    """
    if method.upper() in ("GET", "HEAD"):
        return status in _SAFE_RETRY_STATUSES
    return status == _WRITE_RETRY_STATUS


def parse_retry_after(value: str | None, *, now: datetime) -> float | None:
    """Parse a Retry-After header into seconds, or None to use backoff.

    Accepts a non-negative finite delay-seconds value or an HTTP-date relative
    to UTC. Negative, malformed and non-finite numbers are unusable; a past
    date means zero delay. A date without a timezone is treated as UTC.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return seconds

    try:
        target = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    reference = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return max(0.0, (target - reference).total_seconds())


def _is_tls_verification_failure(exc: BaseException) -> bool:
    """Walk an exception chain for an SSL certificate verification failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        current = current.__cause__ or current.__context__
    return False


def retryable_exception(method: str, exc: httpx.TransportError) -> bool:
    """Whether ``method`` may retry a transport ``exc``.

    Only GET/HEAD retry transport failures, and only the transient remote
    classes named in the policy; TLS verification failures never retry.
    """
    if method.upper() not in ("GET", "HEAD"):
        return False
    if _is_tls_verification_failure(exc):
        return False
    return isinstance(exc, _TRANSIENT_TRANSPORT_ERRORS)


class GitLabRequestError(ProviderError):
    """A GitLab request exhausted its retries, deadline or terminal transport."""

    def __init__(self, reason: str, attempts: int, status_code: int | None = None) -> None:
        self.reason = reason
        self.attempts = attempts
        self.status_code = status_code
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"GitLab request failed after {attempts} attempt(s): {reason}{suffix}")


# --------------------------------------------------------------------------
# Pooled runtime and shared throttle executor
# --------------------------------------------------------------------------

_MAX_ACTIVE_REQUESTS = 8
_MAX_KEEPALIVE_CONNECTIONS = 8
_KEEPALIVE_EXPIRY_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 10.0
_WRITE_TIMEOUT_SECONDS = 30.0
_POOL_TIMEOUT_SECONDS = 5.0
_DEFAULT_READ_TIMEOUT_SECONDS = 30.0


def _monotonic() -> float:
    """Monotonic clock for deadlines and cooldowns; patchable in tests."""
    return asyncio.get_running_loop().time()


async def _sleep(seconds: float) -> None:
    """Sleep helper kept at module scope so tests can inject a fake."""
    await asyncio.sleep(seconds)


def _uniform(low: float, high: float) -> float:
    """Uniform jitter sample; patchable in tests."""
    return random.uniform(low, high)


def _utcnow() -> datetime:
    """UTC clock for HTTP-date parsing; patchable in tests."""
    return datetime.now(UTC)


def _default_client_factory() -> httpx.AsyncClient:
    """Build the pooled client used when no factory is injected."""
    return httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT_SECONDS,
            read=_DEFAULT_READ_TIMEOUT_SECONDS,
            write=_WRITE_TIMEOUT_SECONDS,
            pool=_POOL_TIMEOUT_SECONDS,
        ),
        limits=httpx.Limits(
            max_connections=_MAX_ACTIVE_REQUESTS,
            max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=_KEEPALIVE_EXPIRY_SECONDS,
        ),
    )


def _backoff_delay(attempt: int, policy: GitLabRetryPolicy) -> float:
    """Equal-jitter exponential backoff: uniform in ``[cap / 2, cap]``."""
    cap = min(policy.backoff_cap_seconds, float(2 ** (attempt - 1)))
    return _uniform(cap / 2.0, cap)


@dataclass
class _EndpointState:
    """Per API-origin/token pool state: request slots, client and shared gate."""

    semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(_MAX_ACTIVE_REQUESTS)
    )
    cooldown_not_before: float = 0.0
    client_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class _ApiTarget:
    scheme: str
    host: str
    port: int
    root: str
    origin: str

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}{self.root}/"


def _parse_api_url(api_url: str) -> _ApiTarget:
    """Normalize the configured API URL into scheme/host/port/root components."""
    try:
        parts = urlsplit(api_url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise GitLabRequestError("invalid GitLab API URL", 0) from exc
    if scheme not in ("http", "https") or not host:
        raise GitLabRequestError("invalid GitLab API URL", 0)
    effective_port = port or (443 if scheme == "https" else 80)
    normalized_host = host.lower()
    display_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    return _ApiTarget(
        scheme=scheme,
        host=normalized_host,
        port=effective_port,
        root=parts.path.rstrip("/"),
        origin=f"{scheme}://{display_host}:{effective_port}",
    )


def _resolve_target(url: str, api: _ApiTarget) -> str:
    """Validate an outgoing target and resolve relative URLs against the root.

    Credentials are only attached after this returns, so targets outside the
    configured origin or API root never receive a request.
    """
    try:
        joined = urljoin(api.base_url, str(url))
        # HTTPX canonicalizes dot segments when it builds the request. Apply
        # the same URL semantics before validating so origin/root checks inspect
        # the exact target that will be dispatched.
        candidate = str(httpx.URL(joined))
        parts = urlsplit(candidate)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
        userinfo = (parts.username, parts.password)
    except ValueError as exc:
        raise GitLabRequestError("invalid GitLab API target", 0) from exc
    if scheme not in ("http", "https") or not host or any(userinfo):
        raise GitLabRequestError("invalid GitLab API target", 0)
    effective_port = port or (443 if scheme == "https" else 80)
    if scheme != api.scheme or host.lower() != api.host or effective_port != api.port:
        raise GitLabRequestError("GitLab API target outside configured origin", 0)
    target_root = parts.path.rstrip("/")
    if api.root and not (target_root == api.root or target_root.startswith(api.root + "/")):
        raise GitLabRequestError("GitLab API target outside configured API root", 0)
    return candidate


def _validate_file_value(value: Any) -> None:
    """Recursively reject multipart payloads a retry could not replay.

    HTTPX rewinds seekable file objects on every render, so those replay
    unchanged. Streams without a working ``seek`` would be consumed by the
    first attempt and must be rejected before dispatch.
    """
    if isinstance(value, (bytes, bytearray, str)):
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_file_value(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_file_value(item)
        return
    if hasattr(value, "read"):
        seekable = getattr(value, "seekable", None)
        if not callable(seekable):
            raise GitLabRequestError("non-replayable request body", 0)
        try:
            rewindable = bool(seekable())
        except Exception as exc:
            raise GitLabRequestError("non-replayable request body", 0) from exc
        if not rewindable:
            raise GitLabRequestError("non-replayable request body", 0)
        try:
            value.seek(0)
        except Exception as exc:
            raise GitLabRequestError("non-replayable request body", 0) from exc
        return
    raise GitLabRequestError("non-replayable request body", 0)


def _validate_replayable_body(body_kwargs: dict[str, Any]) -> None:
    """Reject streaming bodies that cannot be replayed for automatic retry."""
    content = body_kwargs.get("content")
    if content is not None and not isinstance(content, (bytes, bytearray, str)):
        raise GitLabRequestError("non-replayable request body", 0)
    data = body_kwargs.get("data")
    if data is not None and not isinstance(data, (dict, list, tuple)):
        raise GitLabRequestError("non-replayable request body", 0)
    files = body_kwargs.get("files")
    if files is not None:
        if not isinstance(files, (dict, list, tuple)):
            raise GitLabRequestError("non-replayable request body", 0)
        _validate_file_value(files)


def _extend_cooldown(state: _EndpointState, not_before: float) -> None:
    """Record a shared not-before instant, never shortening an existing gate."""
    if not_before > state.cooldown_not_before:
        state.cooldown_not_before = not_before


async def _close_for_retry(response: httpx.Response) -> None:
    """Release a retried response's connection before any wait."""
    try:
        await response.aclose()
    except Exception:
        logger.debug("Failed to close a retried GitLab response", exc_info=True)


async def _wait_for_gate(state: _EndpointState, deadline: float, attempts: int) -> None:
    """Sleep outside any permit until the shared cooldown has elapsed."""
    while True:
        now = _monotonic()
        not_before = state.cooldown_not_before
        if now >= not_before:
            return
        remaining = deadline - now
        if remaining <= 0:
            raise GitLabRequestError("overall deadline exceeded", attempts)
        delay = not_before - now
        if delay > remaining:
            raise GitLabRequestError("shared cooldown exceeds remaining deadline", attempts)
        try:
            async with asyncio.timeout(remaining):
                await _sleep(delay)
        except TimeoutError:
            raise GitLabRequestError("overall deadline exceeded", attempts) from None


async def _acquire_permit(state: _EndpointState, deadline: float, attempts: int) -> None:
    """Acquire one of the eight per-endpoint request slots within the deadline."""
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise GitLabRequestError("overall deadline exceeded", attempts)
    try:
        async with asyncio.timeout(remaining):
            await state.semaphore.acquire()
    except TimeoutError:
        raise GitLabRequestError(
            "overall deadline exceeded while waiting for a request slot", attempts
        ) from None


async def _sleep_bounded(
    delay: float,
    deadline: float,
    attempts: int,
    status_code: int | None = None,
) -> None:
    """Sleep a retry delay without ever exceeding the overall deadline."""
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise GitLabRequestError("overall deadline exceeded", attempts, status_code)
    if delay > remaining:
        raise GitLabRequestError("retry delay exceeds remaining deadline", attempts, status_code)
    try:
        async with asyncio.timeout(remaining):
            await _sleep(delay)
    except TimeoutError:
        raise GitLabRequestError("overall deadline exceeded", attempts, status_code) from None


async def _enter_endpoint(state: _EndpointState, deadline: float, attempts: int) -> None:
    """Wait for the gate, take a permit, then re-check the gate.

    If a sibling extended the cooldown after this task passed its first gate
    check, the permit is released and the task starts over instead of sending.
    """
    while True:
        await _wait_for_gate(state, deadline, attempts)
        await _acquire_permit(state, deadline, attempts)
        if _monotonic() >= state.cooldown_not_before:
            return
        state.semaphore.release()


class GitLabHttpRuntime:
    """Own pooled clients, request slots and shared throttle cooldowns."""

    def __init__(
        self,
        *,
        policy: GitLabRetryPolicy | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._policy = policy or GitLabRetryPolicy()
        self._client_factory = client_factory or _default_client_factory
        self._states: dict[tuple[str, str, int, str, str], _EndpointState] = {}
        self._closed = False
        self._pending_clients: list[httpx.AsyncClient] = []
        self._close_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _state_for(self, token_digest: str, api: _ApiTarget) -> _EndpointState:
        key = (api.scheme, api.host, api.port, api.root, token_digest)
        state = self._states.get(key)
        if state is None:
            state = _EndpointState()
            self._states[key] = state
        return state

    async def _ensure_client(self, state: _EndpointState) -> httpx.AsyncClient:
        if state.client is not None:
            return state.client
        async with state.client_lock:
            if state.client is None:
                if self._closed:
                    raise RuntimeError("GitLab HTTP runtime is closed")
                state.client = self._client_factory()
        assert state.client is not None
        return state.client

    async def _send(
        self,
        client: httpx.AsyncClient,
        *,
        method: str,
        target: str,
        headers: dict[str, str],
        body_kwargs: dict[str, Any],
        read_timeout: float,
        deadline: float,
        attempts: int,
    ) -> httpx.Response:
        request = client.build_request(
            method,
            target,
            headers=headers,
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT_SECONDS,
                read=read_timeout,
                write=_WRITE_TIMEOUT_SECONDS,
                pool=_POOL_TIMEOUT_SECONDS,
            ),
            **body_kwargs,
        )
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise GitLabRequestError("overall deadline exceeded", attempts)
        try:
            async with asyncio.timeout(remaining):
                return await client.send(request, follow_redirects=False, stream=False)
        except TimeoutError:
            raise GitLabRequestError("overall deadline exceeded", attempts) from None

    @staticmethod
    def _record_503_cooldown(state: _EndpointState, response: httpx.Response) -> None:
        """Honor a valid 503 Retry-After as shared service-wide guidance."""
        delay = parse_retry_after(response.headers.get("retry-after"), now=_utcnow())
        if delay is not None:
            _extend_cooldown(state, _monotonic() + delay)

    async def request(
        self,
        token: str,
        api_url: str,
        method: str,
        url: str,
        *,
        read_timeout: float = _DEFAULT_READ_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute one logical GitLab request with the shared retry policy."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("GitLabHttpRuntime cannot be used from a different event loop")
        if self._closed:
            raise RuntimeError("GitLab HTTP runtime is closed")

        # The deadline starts before any queueing, attempt or sleep.
        deadline = _monotonic() + self._policy.deadline_seconds
        attempts = 0

        api = _parse_api_url(api_url)
        target = _resolve_target(url, api)
        body_kwargs = dict(kwargs)
        headers = dict(body_kwargs.pop("headers", None) or {})
        _validate_replayable_body(body_kwargs)
        headers["PRIVATE-TOKEN"] = token

        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        state = self._state_for(digest, api)
        client = await self._ensure_client(state)

        if self._policy.max_attempts <= 0:
            raise GitLabRequestError("retry policy allows no attempts", 0)
        if self._policy.deadline_seconds <= 0:
            raise GitLabRequestError("overall deadline exceeded", 0)

        display = f"{api.origin}{urlsplit(target).path}"
        while attempts < self._policy.max_attempts:
            attempts += 1
            logger.debug(
                "GitLab %s %s attempt %d/%d",
                method,
                display,
                attempts,
                self._policy.max_attempts,
            )
            await _enter_endpoint(state, deadline, attempts)
            sleep_delay: float | None = None
            failure: GitLabRequestError | None = None
            status_code: int | None = None
            try:
                try:
                    response = await self._send(
                        client,
                        method=method,
                        target=target,
                        headers=headers,
                        body_kwargs=body_kwargs,
                        read_timeout=read_timeout,
                        deadline=deadline,
                        attempts=attempts,
                    )
                except GitLabRequestError:
                    raise
                except httpx.TransportError as exc:
                    transient = retryable_exception(method, exc)
                    if transient and attempts >= self._policy.max_attempts:
                        failure = GitLabRequestError(
                            "transient transport failure after "
                            f"{attempts} attempt(s): {type(exc).__name__}",
                            attempts,
                        )
                    elif transient:
                        sleep_delay = _backoff_delay(attempts, self._policy)
                        logger.info(
                            "GitLab %s %s attempt %d transport retry in %.3fs: %s",
                            method,
                            display,
                            attempts,
                            sleep_delay,
                            type(exc).__name__,
                        )
                    else:
                        failure = GitLabRequestError(
                            f"terminal transport failure: {type(exc).__name__}",
                            attempts,
                        )
                except Exception as exc:
                    raise GitLabRequestError(
                        f"terminal transport failure: {type(exc).__name__}", attempts
                    ) from exc
                else:
                    if not retryable_status(method, response.status_code):
                        if response.status_code == 503:
                            self._record_503_cooldown(state, response)
                        return response

                    status_code = response.status_code
                    header_delay = parse_retry_after(
                        response.headers.get("retry-after"), now=_utcnow()
                    )
                    delay = (
                        _backoff_delay(attempts, self._policy)
                        if header_delay is None
                        else header_delay
                    )
                    if status_code == 429 or (status_code == 503 and header_delay is not None):
                        _extend_cooldown(state, _monotonic() + delay)
                    # Close the retried response before waiting; the permit is
                    # released by the finally below, never held across a sleep.
                    await _close_for_retry(response)
                    if attempts >= self._policy.max_attempts:
                        failure = GitLabRequestError(
                            f"exhausted retries (HTTP {status_code})",
                            attempts,
                            status_code,
                        )
                    else:
                        sleep_delay = delay
                        logger.info(
                            "GitLab %s %s attempt %d status %d retry in %.3fs",
                            method,
                            display,
                            attempts,
                            status_code,
                            delay,
                        )
            finally:
                state.semaphore.release()

            if failure is not None:
                raise failure
            if sleep_delay is not None:
                await _sleep_bounded(sleep_delay, deadline, attempts, status_code)

        raise GitLabRequestError("exhausted retries", attempts)

    async def aclose(self) -> None:
        """Close pooled clients once; interrupted cleanup resumes on the next call."""
        if self._closed and not self._pending_clients:
            return
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("GitLabHttpRuntime cannot be closed from a different event loop")
        self._closed = True
        async with self._close_lock:
            if not self._pending_clients:
                self._pending_clients.extend(
                    state.client for state in self._states.values() if state.client is not None
                )
                for state in self._states.values():
                    state.client = None
            index = 0
            while index < len(self._pending_clients):
                client = self._pending_clients[index]
                cancelled = False
                try:
                    # Shield so an interrupted teardown leaves the client
                    # pending for the next aclose() instead of losing it.
                    await asyncio.shield(client.aclose())
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                except Exception:
                    logger.debug("Failed to close a pooled GitLab client", exc_info=True)
                finally:
                    if not cancelled:
                        del self._pending_clients[index]


_RUNTIME_CONTEXT: ContextVar[GitLabHttpRuntime | None] = ContextVar(
    "mira_gitlab_http_runtime", default=None
)


@asynccontextmanager
async def gitlab_http_scope(
    runtime: GitLabHttpRuntime | None = None,
):
    """Bind a runtime for the scope, owning (and closing) only a new one."""
    if runtime is None:
        runtime = GitLabHttpRuntime()
        owned = True
    else:
        owned = False
    token = _RUNTIME_CONTEXT.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME_CONTEXT.reset(token)
        if owned:
            await runtime.aclose()


async def request_gitlab(
    token: str,
    api_url: str,
    method: str,
    url: str,
    *,
    read_timeout: float = _DEFAULT_READ_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> httpx.Response:
    """Resolve the bound runtime, or use a temporary one for this request."""
    runtime = _RUNTIME_CONTEXT.get()
    if runtime is not None:
        return await runtime.request(
            token, api_url, method, url, read_timeout=read_timeout, **kwargs
        )
    runtime = GitLabHttpRuntime()
    try:
        return await runtime.request(
            token, api_url, method, url, read_timeout=read_timeout, **kwargs
        )
    finally:
        await runtime.aclose()
