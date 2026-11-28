"""Pagination ``Link`` header parsing shared by platform callers."""

from __future__ import annotations

import re

import httpx

_LINK_PART = re.compile(r'\s*<([^<>]+)>\s*;\s*rel=(?:"([^"]+)"|([^;\s]+))\s*')


def parse_next_link(link_header: str) -> str | None:
    """Extract the ``rel="next"`` URL from a ``Link`` header."""
    if not link_header:
        return None
    next_url: str | None = None
    for part in link_header.split(","):
        match = _LINK_PART.fullmatch(part)
        if not match:
            raise ValueError("Invalid pagination link")
        if "next" in (match[2] or match[3]).split():
            if next_url is not None:
                raise ValueError("Multiple pagination next links")
            next_url = match[1]
    return next_url


def same_origin_next_link(current_url: str, link_header: str) -> str | None:
    """Like :func:`parse_next_link`, but refuse cross-origin redirect targets.

    Pagination URLs come from an untrusted upstream response while the request
    carries credentials, so following them off-site would leak the token.
    """
    next_url = parse_next_link(link_header)
    if next_url is None:
        return None
    current = httpx.URL(current_url)
    target = httpx.URL(next_url)
    if target.userinfo or (target.scheme, target.host, target.port) != (
        current.scheme,
        current.host,
        current.port,
    ):
        raise ValueError("Pagination link leaves the API origin")
    return next_url
