"""Tests for pagination Link-header parsing and origin checks."""

from __future__ import annotations

import pytest

from mira.platforms.links import parse_next_link, same_origin_next_link


def test_parse_next_link_extracts_url() -> None:
    header = '<https://gitlab.com/api/v4/projects?page=2>; rel="next"'
    assert parse_next_link(header) == "https://gitlab.com/api/v4/projects?page=2"


def test_parse_next_link_without_next_returns_none() -> None:
    assert parse_next_link('<https://gitlab.com/api/v4/projects>; rel="prev"') is None
    assert parse_next_link("") is None


def test_parse_next_link_rejects_multiple_next_links() -> None:
    header = '<https://gitlab.com/a?page=2>; rel="next", <https://gitlab.com/a?page=3>; rel="next"'
    with pytest.raises(ValueError, match="Multiple"):
        parse_next_link(header)


def test_parse_next_link_ignores_extra_parameters_and_odd_parts() -> None:
    header = (
        "garbage; not-a-link, "
        '<https://gitlab.com/a?page=2>; rel="next"; type="application/json", '
        '<https://gitlab.com/a?page=1>; rel="prev"'
    )
    assert parse_next_link(header) == "https://gitlab.com/a?page=2"


def test_same_origin_rejects_malformed_target_url() -> None:
    # An invalid port makes httpx.URL raise InvalidURL, which must surface as ValueError.
    header = '<https://gitlab.com:bad/api/v4/projects?page=2>; rel="next"'
    with pytest.raises(ValueError, match="Invalid pagination link"):
        same_origin_next_link("https://gitlab.com/api/v4/projects", header)


def test_same_origin_accepts_same_host() -> None:
    header = '<https://gitlab.com/api/v4/projects?page=2>; rel="next"'
    assert same_origin_next_link("https://gitlab.com/api/v4/projects", header) == (
        "https://gitlab.com/api/v4/projects?page=2"
    )


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example/api/v4/projects?page=2",
        "http://gitlab.com/api/v4/projects?page=2",
        "https://user@gitlab.com/api/v4/projects?page=2",
    ],
)
def test_same_origin_rejects_foreign_targets(target: str) -> None:
    header = f'<{target}>; rel="next"'
    with pytest.raises(ValueError, match="origin"):
        same_origin_next_link("https://gitlab.com/api/v4/projects", header)
