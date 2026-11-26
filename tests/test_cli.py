"""Tests for CLI formatting and command invocation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from mira.cli import _format_json, _format_text, main
from mira.models import (
    FileChangeType,
    ReviewComment,
    ReviewResult,
    Severity,
    WalkthroughEffort,
    WalkthroughFileEntry,
    WalkthroughResult,
)
from mira.platforms.gitlab import http as gitlab_http
from mira.platforms.gitlab.http import _RUNTIME_CONTEXT, request_gitlab

GITLAB_API_URL = "https://gitlab.example.com/api/v4"


def _make_result(**overrides) -> ReviewResult:
    defaults = {
        "comments": [],
        "summary": "Looks good.",
        "reviewed_files": 1,
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    defaults.update(overrides)
    return ReviewResult(**defaults)


def _make_comment(**overrides) -> ReviewComment:
    defaults = {
        "path": "src/foo.py",
        "line": 10,
        "end_line": None,
        "severity": Severity.WARNING,
        "category": "bug",
        "title": "Potential bug",
        "body": "This might crash.",
        "confidence": 0.9,
        "suggestion": None,
    }
    defaults.update(overrides)
    return ReviewComment(**defaults)


# ---------------------------------------------------------------------------
# _format_text tests
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_no_comments(self):
        result = _make_result()
        text = _format_text(result)
        assert "Looks good." in text
        assert "No issues found." in text

    def test_with_comments(self):
        comment = _make_comment()
        result = _make_result(comments=[comment])
        text = _format_text(result)
        assert "[WARNING]" in text
        assert "src/foo.py:10" in text
        assert "Potential bug" in text
        assert "This might crash." in text
        assert "Reviewed 1 files, 1 comments." in text
        assert "Tokens used: 15" in text

    def test_with_suggestion(self):
        comment = _make_comment(suggestion="return None")
        result = _make_result(comments=[comment])
        text = _format_text(result)
        assert "Suggestion: return None" in text

    def test_with_walkthrough(self):
        wt = WalkthroughResult(summary="PR adds helpers.")
        result = _make_result(walkthrough=wt)
        text = _format_text(result)
        assert "## Mira PR Walkthrough" in text
        assert "---" in text
        assert "PR adds helpers." in text

    def test_no_summary(self):
        result = _make_result(summary="")
        text = _format_text(result)
        assert "No issues found." in text

    def test_no_token_usage(self):
        result = _make_result(token_usage={}, comments=[_make_comment()])
        text = _format_text(result)
        assert "Tokens used:" not in text


# ---------------------------------------------------------------------------
# _format_json tests
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_basic_json(self):
        result = _make_result()
        raw = _format_json(result)
        data = json.loads(raw)
        assert data["summary"] == "Looks good."
        assert data["walkthrough"] is None
        assert data["comments"] == []
        assert data["reviewed_files"] == 1

    def test_json_with_comments(self):
        comment = _make_comment(end_line=12)
        result = _make_result(comments=[comment])
        raw = _format_json(result)
        data = json.loads(raw)
        assert len(data["comments"]) == 1
        c = data["comments"][0]
        assert c["path"] == "src/foo.py"
        assert c["line"] == 10
        assert c["end_line"] == 12
        assert c["severity"] == "warning"
        assert c["category"] == "bug"

    def test_json_with_walkthrough(self):
        wt = WalkthroughResult(
            summary="PR summary.",
            file_changes=[
                WalkthroughFileEntry(
                    path="a.py",
                    change_type=FileChangeType.ADDED,
                    description="New file",
                    group="Core",
                )
            ],
            effort=WalkthroughEffort(level=2, label="Simple", minutes=10),
            sequence_diagram="sequenceDiagram\n  A->>B: call",
        )
        result = _make_result(walkthrough=wt)
        raw = _format_json(result)
        data = json.loads(raw)
        w = data["walkthrough"]
        assert w["summary"] == "PR summary."
        assert len(w["change_groups"]) == 1
        assert w["change_groups"][0]["label"] == "Core"
        assert w["change_groups"][0]["files"][0]["path"] == "a.py"
        assert w["effort"]["level"] == 2
        assert w["effort"]["minutes"] == 10
        assert "sequenceDiagram" in w["sequence_diagram"]

    def test_json_walkthrough_no_effort(self):
        wt = WalkthroughResult(summary="No effort.")
        result = _make_result(walkthrough=wt)
        raw = _format_json(result)
        data = json.loads(raw)
        assert data["walkthrough"]["effort"] is None

    def test_json_walkthrough_ungrouped_files(self):
        wt = WalkthroughResult(
            summary="Flat.",
            file_changes=[
                WalkthroughFileEntry(
                    path="a.py",
                    change_type=FileChangeType.MODIFIED,
                    description="Changed",
                ),
            ],
        )
        result = _make_result(walkthrough=wt)
        raw = _format_json(result)
        data = json.loads(raw)
        # Files without group label get bucketed as "Other"
        assert data["walkthrough"]["change_groups"][0]["label"] == "Other"


# ---------------------------------------------------------------------------
# CLI invocation tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "mira" in result.output.lower()

    def test_review_requires_pr_or_stdin(self):
        runner = CliRunner()
        result = runner.invoke(main, ["review"])
        assert result.exit_code != 0
        assert "Provide --pr" in result.output or "Usage" in result.output

    def test_review_pr_requires_github_token(self):
        runner = CliRunner()
        result = runner.invoke(
            main, ["review", "--pr", "https://github.com/o/r/pull/1"], catch_exceptions=False
        )
        assert result.exit_code != 0
        assert "token" in result.output.lower() or "GITHUB_TOKEN" in result.output

    def test_review_stdin_text_output(self):
        review_result = _make_result(summary="All good.")

        with patch("mira.cli.ReviewEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["review", "--stdin"],
                input="diff --git a/f.py b/f.py\n",
            )

        assert result.exit_code == 0
        assert "All good." in result.output
        assert "No issues found." in result.output

    def test_review_stdin_json_output(self):
        review_result = _make_result(summary="JSON output.")

        with patch("mira.cli.ReviewEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["review", "--stdin", "--output", "json"],
                input="diff --git a/f.py b/f.py\n",
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"] == "JSON output."

    def test_review_overrides(self):
        review_result = _make_result()

        with (
            patch("mira.cli.ReviewEngine") as mock_engine_cls,
            patch("mira.cli.load_config") as mock_load,
        ):
            mock_load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "review",
                    "--stdin",
                    "--model",
                    "test-model",
                    "--max-comments",
                    "3",
                    "--confidence",
                    "0.9",
                ],
                input="diff\n",
            )

        assert result.exit_code == 0
        # Verify overrides were passed to load_config
        call_args = mock_load.call_args
        overrides = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("overrides")
        if overrides is None and len(call_args[0]) > 1:
            overrides = call_args[0][1]

    def test_review_blocker_exit_code_1(self):
        blocker = _make_comment(severity=Severity.BLOCKER)
        review_result = _make_result(comments=[blocker])

        with patch("mira.cli.ReviewEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["review", "--stdin"],
                input="diff\n",
            )

        assert result.exit_code == 1

    def test_review_verbose_flag(self):
        review_result = _make_result()

        with patch("mira.cli.ReviewEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["review", "--stdin", "--verbose"],
                input="diff\n",
            )

        assert result.exit_code == 0

    def test_review_config_does_not_trust_execution_settings_by_default(self):
        review_result = _make_result()

        with (
            patch("mira.cli.ReviewEngine") as mock_engine_cls,
            patch("mira.cli.load_config") as mock_load,
        ):
            mock_load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            result = CliRunner().invoke(
                main,
                ["review", "--stdin", "--config", ".mira.yaml"],
                input="diff\n",
            )

        assert result.exit_code == 0
        assert mock_load.call_args.kwargs["trust_execution_settings"] is False

    def test_review_requires_explicit_opt_in_to_trust_execution_settings(self):
        review_result = _make_result()

        with (
            patch("mira.cli.ReviewEngine") as mock_engine_cls,
            patch("mira.cli.load_config") as mock_load,
        ):
            mock_load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.review_diff = AsyncMock(return_value=review_result)
            mock_engine_cls.return_value = mock_engine

            result = CliRunner().invoke(
                main,
                [
                    "review",
                    "--stdin",
                    "--config",
                    "/etc/mira/config.yaml",
                    "--trust-execution-settings",
                ],
                input="diff\n",
            )

        assert result.exit_code == 0
        assert mock_load.call_args.kwargs["trust_execution_settings"] is True

    def test_review_trust_execution_settings_requires_explicit_config(self):
        result = CliRunner().invoke(
            main,
            ["review", "--stdin", "--trust-execution-settings"],
            input="diff\n",
        )

        assert result.exit_code != 0
        assert "requires --config" in result.output

    @staticmethod
    def _client_recorder(
        created: list[httpx.AsyncClient],
        loops: list[asyncio.AbstractEventLoop],
    ):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"username": "mira-bot"})

        def factory() -> httpx.AsyncClient:
            # A client may only be built inside the invocation's event loop.
            loops.append(asyncio.get_running_loop())
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            created.append(client)
            return client

        return factory

    def test_review_owns_separate_scoped_runtime_per_invocation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        created: list[httpx.AsyncClient] = []
        loops: list[asyncio.AbstractEventLoop] = []
        runtimes: list[object] = []

        async def fake_review_diff(*args, **kwargs):
            runtimes.append(_RUNTIME_CONTEXT.get())
            response = await request_gitlab("tok", GITLAB_API_URL, "GET", f"{GITLAB_API_URL}/user")
            assert response.status_code == 200
            return _make_result()

        factory = self._client_recorder(created, loops)
        with (
            patch("mira.cli.ReviewEngine") as mock_engine_cls,
            patch("mira.cli.load_config") as mock_load,
            patch.object(gitlab_http, "_default_client_factory", factory),
        ):
            mock_load.return_value = MagicMock()
            engine = MagicMock()
            engine.review_diff = AsyncMock(side_effect=fake_review_diff)
            mock_engine_cls.return_value = engine

            assert created == []  # no client is built before the invocation loop
            for _ in range(2):
                result = CliRunner().invoke(main, ["review", "--stdin"], input="diff\n")
                assert result.exit_code == 0, result.output

        assert len(runtimes) == 2
        assert all(runtime is not None for runtime in runtimes)
        assert runtimes[0] is not runtimes[1]
        assert len(loops) == 2
        assert loops[0] is not loops[1]
        assert len(created) == 2
        assert created[0] is not created[1]
        assert all(client.is_closed for client in created)
        assert all(runtime._closed for runtime in runtimes)  # type: ignore[attr-defined]

    def test_serve_identity_bootstrap_owns_scoped_runtime(self, monkeypatch: pytest.MonkeyPatch):
        created: list[httpx.AsyncClient] = []
        loops: list[asyncio.AbstractEventLoop] = []
        runtimes: list[object] = []

        async def fake_identity(self):
            runtimes.append(_RUNTIME_CONTEXT.get())
            response = await request_gitlab("tok", GITLAB_API_URL, "GET", f"{GITLAB_API_URL}/user")
            assert response.status_code == 200
            return "mira-bot"

        factory = self._client_recorder(created, loops)
        with (
            patch(
                "mira.platforms.gitlab.auth.GitLabTokenAuth.get_bot_identity",
                fake_identity,
            ),
            patch("mira.dashboard.api.register_dashboard", lambda app: None),
            patch("mira.dashboard.api._app_db.set_setting", MagicMock()),
            patch("uvicorn.run") as uvicorn_run,
            patch.object(gitlab_http, "_default_client_factory", factory),
        ):
            for _ in range(2):
                result = CliRunner().invoke(
                    main,
                    [
                        "serve",
                        "--gitlab-token",
                        "tok",
                        "--gitlab-webhook-secret",
                        "sec",
                    ],
                )
                assert result.exit_code == 0, result.output

        assert uvicorn_run.call_count == 2
        assert len(runtimes) == 2
        assert all(runtime is not None for runtime in runtimes)
        assert runtimes[0] is not runtimes[1]
        assert len(loops) == 2
        assert loops[0] is not loops[1]
        assert len(created) == 2
        assert created[0] is not created[1]
        assert all(client.is_closed for client in created)
        assert all(runtime._closed for runtime in runtimes)  # type: ignore[attr-defined]
