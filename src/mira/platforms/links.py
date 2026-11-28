"""Pagination ``Link`` header parsing shared by platform callers."""

from __future__ import annotations

import re

import httpx

_LINK_PART = re.compile(r"\s*<([^<>]+)>\s*;\s*(.*?)\s*")


def _relation_values(params: str) -> list[str]:
    """Return every ``rel`` token from a Link parameter list."""
    values: list[str] = []
    for param in params.split(";"):
        key, separator, value = param.partition("=")
        if not separator or key.strip().lower() != "rel":
            continue
        value = value.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        values.extend(value.split())
    return values


def parse_next_link(link_header: str) -> str | None:
    """Extract the ``rel="next"`` URL from a ``Link`` header.

    Additional link parameters are ignored. A header that contains no link
    target at all is rejected; unrelated malformed parts are skipped so one
    odd parameter cannot silently truncate a paginated listing.
    """
    if not link_header:
        return None
    next_url: str | None = None
    parsed_any = False
    for part in link_header.split(","):
        match = _LINK_PART.fullmatch(part)
        if not match:
            continue
        parsed_any = True
        if "next" in _relation_values(match.group(2)):
            if next_url is not None:
                raise ValueError("Multiple pagination next links")
            next_url = match.group(1)
    if not parsed_any:
        raise ValueError("Invalid pagination link")
    return next_url


def same_origin_next_link(current_url: str, link_header: str) -> str | None:
    """Like :func:`parse_next_link`, but refuse cross-origin redirect targets.

    Pagination URLs come from an untrusted upstream response while the request
    carries credentials, so following them off-site would leak the token.
    """
    next_url = parse_next_link(link_header)
    if next_url is None:
        return None
    try:
        current = httpx.URL(current_url)
        target = httpx.URL(next_url)
    except httpx.InvalidURL as exc:
        raise ValueError("Invalid pagination link") from exc
    if target.userinfo or (target.scheme, target.host, target.port) != (
        current.scheme,
        current.host,
        current.port,
    ):
        raise ValueError("Pagination link leaves the API origin")
    return next_url
