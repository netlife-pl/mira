"""GitLab HTTP policy: retry decisions and server-directed delays.

Pure helpers only — no HTTP calls or global client state. The runtime executor
built on top of these consumes the same policy for every logical request.
"""

from __future__ import annotations

import math
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from mira.exceptions import ProviderError


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
