"""Tests for GitLabProvider (mira.providers.gitlab)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import pytest

from mira.exceptions import ProviderError
from mira.models import PRInfo, ReviewComment, ReviewResult, Severity
from mira.platforms.gitlab import http as gitlab_http
from mira.platforms.gitlab.http import (
    GitLabHttpRuntime,
    GitLabRequestError,
    gitlab_http_scope,
)
from mira.providers import create_provider
from mira.providers.gitlab import GitLabProvider, _build_unified_diff, parse_mr_url


class _FakeResp:
    def __init__(self, status=200, json_data=None, text="", headers=None):
        self.status_code = status
        self._json = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aclose(self):
        return None


class _RoutedRequest:
    """The slice of httpx.Request the executor's client protocol consumes."""

    def __init__(self, method: str, url: str, headers: dict, kwargs: dict) -> None:
        self.method = method
        self.url = url
        self.headers = headers
        self.kwargs = kwargs


class _RoutedClient:
    """Injected runtime client translating build_request/send back to a fake."""

    def __init__(self, handler) -> None:
        self._handler = handler

    def build_request(self, method, url, *, headers=None, timeout=None, **kwargs):
        return _RoutedRequest(method, str(url), headers or {}, kwargs)

    async def send(self, request, *, follow_redirects=False, stream=False):
        return self._handler(request.method, request.url, headers=request.headers, **request.kwargs)

    async def aclose(self):
        return None


@asynccontextmanager
async def _patch(handler):
    runtime = GitLabHttpRuntime(client_factory=lambda: _RoutedClient(handler))
    async with gitlab_http_scope(runtime):
        try:
            yield runtime
        finally:
            await runtime.aclose()


@asynccontextmanager
async def _mock_scope(handler):
    """Bind a runtime whose real HTTPX client is backed by MockTransport."""
    runtime = GitLabHttpRuntime(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    async with gitlab_http_scope(runtime):
        try:
            yield runtime
        finally:
            await runtime.aclose()


_PR = PRInfo(
    title="t",
    description="d",
    base_branch="main",
    head_branch="feat",
    url="u",
    number=7,
    owner="group/sub",
    repo="proj",
    head_sha="abc",
)


class TestParseMrUrl:
    def test_full_url_nested_group(self):
        owner, repo, iid = parse_mr_url("https://gitlab.com/group/sub/proj/-/merge_requests/42")
        assert owner == "group/sub"
        assert repo == "proj"
        assert iid == 42

    def test_shorthand(self):
        assert parse_mr_url("group/proj!13") == ("group", "proj", 13)

    def test_invalid(self):
        with pytest.raises(ProviderError):
            parse_mr_url("not-a-url")


class TestBuildUnifiedDiff:
    def test_headers_prepended(self):
        changes = [{"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1 +1 @@\n-x\n+y\n"}]
        out = _build_unified_diff(changes)
        assert "--- a/a.py" in out
        assert "+++ b/a.py" in out
        assert "@@ -1 +1 @@" in out

    def test_added_file_uses_devnull(self):
        changes = [
            {
                "old_path": "n.py",
                "new_path": "n.py",
                "new_file": True,
                "diff": "@@ -0,0 +1 @@\n+x\n",
            }
        ]
        out = _build_unified_diff(changes)
        assert "--- /dev/null" in out
        assert "+++ b/n.py" in out
        assert "new file mode" in out

    def test_added_file_parses_as_single_file(self):
        # Regression: a `diff --git` header + /dev/null without a file-mode line
        # makes unidiff emit the file twice (the empty entry then clobbers the
        # real line ranges, so every finding is dropped). The mode line fixes it.
        from mira.core.diff_parser import parse_diff

        changes = [
            {
                "old_path": "n.py",
                "new_path": "n.py",
                "new_file": True,
                "diff": "@@ -0,0 +1,2 @@\n+import os\n+x = 1\n",
            }
        ]
        files = parse_diff(_build_unified_diff(changes)).files
        assert len(files) == 1
        assert files[0].path == "n.py"
        assert len(files[0].hunks) == 1


class TestGetPrInfo:
    @pytest.mark.asyncio
    async def test_maps_iid_and_branches(self):
        def handler(method, url, **kw):
            assert method == "GET"
            assert url.endswith("/projects/g%2Fp/merge_requests/7")
            return _FakeResp(
                json_data={
                    "title": "Add feature",
                    "description": "body",
                    "target_branch": "main",
                    "source_branch": "feat",
                    "web_url": "https://gitlab.com/g/p/-/merge_requests/7",
                    "sha": "deadbeef",
                    "author": {
                        "username": "example-author",
                        "name": "Example Display Name",
                        "avatar_url": "https://example.invalid/avatar.png",
                    },
                }
            )

        async with _patch(handler):
            info = await GitLabProvider("tok").get_pr_info(
                "https://gitlab.com/g/p/-/merge_requests/7"
            )
        assert info.number == 7
        assert info.base_branch == "main"
        assert info.head_branch == "feat"
        assert info.head_sha == "deadbeef"
        assert info.author == "example-author"
        assert info.author_avatar_url == "https://example.invalid/avatar.png"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fields", "expected_username"),
        [
            pytest.param({}, "", id="missing-author"),
            pytest.param({"author": None}, "", id="null-author"),
            pytest.param({"author": {}}, "", id="empty-author"),
            pytest.param({"author": "invalid"}, "", id="non-object-author"),
            pytest.param(
                {"author": {"username": "example-author"}}, "example-author", id="no-avatar"
            ),
            pytest.param(
                {"author": {"username": "example-author", "avatar_url": None}},
                "example-author",
                id="null-avatar",
            ),
            pytest.param(
                {"author": {"username": ["invalid"], "avatar_url": {"invalid": True}}},
                "",
                id="non-string-fields",
            ),
        ],
    )
    async def test_defaults_incomplete_author_metadata(self, fields, expected_username):
        def handler(method, url, **kw):
            return _FakeResp(json_data={"title": "Example MR", **fields})

        async with _patch(handler):
            info = await GitLabProvider("tok").get_pr_info("g/p!7")

        assert info.title == "Example MR"
        assert info.author == expected_username
        assert info.author_avatar_url == ""


class TestPostReview:
    @pytest.mark.asyncio
    async def test_inline_position_payload(self):
        calls = []

        def handler(method, url, **kw):
            calls.append((method, url, kw.get("data")))
            if url.endswith("/changes"):
                return _FakeResp(
                    json_data={
                        "diff_refs": {"base_sha": "b", "start_sha": "s", "head_sha": "h"},
                        "changes": [],
                    }
                )
            return _FakeResp(status=201, json_data={"id": 1})

        result = ReviewResult(
            comments=[
                ReviewComment(
                    path="a.py",
                    line=10,
                    end_line=None,
                    severity=Severity.WARNING,
                    category="bug",
                    title="x",
                    body="y",
                    confidence=0.9,
                )
            ],
            summary="",
            key_issues=[],
        )
        async with _patch(handler):
            await GitLabProvider("tok").post_review(_PR, result)

        disc = next(c for c in calls if c[1].endswith("/discussions"))
        data = disc[2]
        assert data["position[new_line]"] == "10"
        assert data["position[base_sha]"] == "b"
        assert data["position[new_path]"] == "a.py"

    @pytest.mark.asyncio
    async def test_falls_back_to_note_on_400(self):
        calls = []

        def handler(method, url, **kw):
            calls.append((method, url, kw.get("data")))
            if url.endswith("/changes"):
                return _FakeResp(json_data={"diff_refs": {}, "changes": []})
            if url.endswith("/discussions"):
                return _FakeResp(status=400, text="line not in diff")
            return _FakeResp(status=201, json_data={"id": 1})  # /notes

        result = ReviewResult(
            comments=[
                ReviewComment(
                    path="a.py",
                    line=10,
                    end_line=None,
                    severity=Severity.WARNING,
                    category="bug",
                    title="x",
                    body="y",
                    confidence=0.9,
                )
            ],
            summary="",
            key_issues=[],
        )
        async with _patch(handler):
            await GitLabProvider("tok").post_review(_PR, result)

        # The inline 400 should have produced a plain note containing the path.
        notes = [c for c in calls if c[1].endswith("/notes")]
        assert any("a.py:10" in (c[2] or {}).get("body", "") for c in notes)

    @staticmethod
    def _one_comment_result() -> ReviewResult:
        return ReviewResult(
            comments=[
                ReviewComment(
                    path="a.py",
                    line=10,
                    end_line=None,
                    severity=Severity.WARNING,
                    category="bug",
                    title="x",
                    body="y",
                    confidence=0.9,
                )
            ],
            summary="",
            key_issues=[],
        )

    @pytest.mark.asyncio
    async def test_lost_discussion_response_does_not_post_fallback_note(self):
        """An ambiguous POST failure must not trigger a second write.

        The discussion may already have been committed server-side; the plain
        note fallback would duplicate the finding.
        """
        calls = []

        def handler(method, url, **kw):
            calls.append((method, url, kw.get("data")))
            if url.endswith("/changes"):
                return _FakeResp(json_data={"diff_refs": {}, "changes": []})
            if url.endswith("/discussions"):
                raise httpx.ReadTimeout("response lost after commit")
            return _FakeResp(status=201, json_data={"id": 1})  # /notes

        async with _patch(handler):
            with pytest.raises(GitLabRequestError):
                await GitLabProvider("tok").post_review(_PR, self._one_comment_result())

        assert [c for c in calls if c[1].endswith("/notes")] == []
        assert len([c for c in calls if c[1].endswith("/discussions")]) == 1

    @pytest.mark.asyncio
    async def test_server_error_does_not_post_fallback_note(self):
        """Only a positively identified invalid-position 400 falls back."""
        calls = []

        def handler(method, url, **kw):
            calls.append((method, url, kw.get("data")))
            if url.endswith("/changes"):
                return _FakeResp(json_data={"diff_refs": {}, "changes": []})
            if url.endswith("/discussions"):
                return _FakeResp(status=500, text="upstream exploded")
            return _FakeResp(status=201, json_data={"id": 1})  # /notes

        async with _patch(handler):
            with pytest.raises(GitLabRequestError) as excinfo:
                await GitLabProvider("tok").post_review(_PR, self._one_comment_result())

        assert excinfo.value.status_code == 500
        assert [c for c in calls if c[1].endswith("/notes")] == []


class TestCompareDiff:
    @pytest.mark.asyncio
    async def test_builds_diff_from_compare(self):
        def handler(method, url, **kw):
            assert "from=base1" in url and "to=head2" in url
            return _FakeResp(
                json_data={
                    "diffs": [
                        {"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1 +1 @@\n-x\n+y\n"}
                    ]
                }
            )

        async with _patch(handler):
            out = await GitLabProvider("tok").get_compare_diff(_PR, "base1", "head2")
        assert "--- a/a.py" in out and "@@ -1 +1 @@" in out

    @pytest.mark.asyncio
    async def test_identical_shas_returns_empty(self):
        out = await GitLabProvider("tok").get_compare_diff(_PR, "x", "x")
        assert out == ""


class TestResolveThreads:
    @pytest.mark.asyncio
    async def test_puts_resolved_true(self):
        calls = []

        def handler(method, url, **kw):
            calls.append((method, url, kw.get("data")))
            return _FakeResp(status=200, json_data={})

        async with _patch(handler):
            n = await GitLabProvider("tok").resolve_threads(_PR, ["d1", "d2"])
        assert n == 2
        assert all(c[2] == {"resolved": "true"} for c in calls)
        assert all("/discussions/" in c[1] for c in calls)


class TestDiscussionRootBody:
    @pytest.mark.asyncio
    async def test_returns_first_note(self):
        def handler(method, url, **kw):
            assert "/discussions/abc" in url
            return _FakeResp(
                json_data={"notes": [{"body": "original suggestion"}, {"body": "reply"}]}
            )

        async with _patch(handler):
            body = await GitLabProvider("tok").get_discussion_root_body(_PR, "abc")
        assert body == "original suggestion"

    @pytest.mark.asyncio
    async def test_empty_on_error(self):
        def handler(method, url, **kw):
            return _FakeResp(status=404, text="nope")

        async with _patch(handler):
            assert await GitLabProvider("tok").get_discussion_root_body(_PR, "abc") == ""


class TestBotThreadIdentity:
    @pytest.mark.asyncio
    async def test_matches_token_user_not_configured_name(self):
        # The bot posts as the access-token user (project_99_bot_x), which is
        # unrelated to the configured display name "mira". Dedup/resolve must
        # still recognise its own threads.
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "author": {"username": "project_99_bot_x"},
                        "position": {"new_path": "a.py", "new_line": 5},
                        "body": "bot finding",
                        "resolved": False,
                    }
                ],
            },
            {
                "id": "d2",
                "notes": [
                    {
                        "author": {"username": "alice"},
                        "position": {"new_path": "a.py", "new_line": 9},
                        "body": "human note",
                        "resolved": False,
                    }
                ],
            },
        ]

        def handler(method, url, **kw):
            if url.endswith("/user"):
                return _FakeResp(json_data={"username": "project_99_bot_x"})
            return _FakeResp(json_data=discussions)

        async with _patch(handler):
            threads = await GitLabProvider("tok").get_unresolved_bot_threads(_PR, "mira")
        assert [t.thread_id for t in threads] == ["d1"]


class TestGetFileContent:
    @pytest.mark.asyncio
    async def test_returns_raw_body(self):
        def handler(method, url, **kw):
            assert "/repository/files/" in url and "/raw?ref=" in url
            return _FakeResp(text="x = 1\n")

        async with _patch(handler):
            body = await GitLabProvider("tok").get_file_content(_PR, "a.py", "feat")
        assert body == "x = 1\n"

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self):
        def handler(method, url, **kw):
            return _FakeResp(status=404, text="not found")

        async with _patch(handler):
            assert await GitLabProvider("tok").get_file_content(_PR, "gone.py", "feat") == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_terminal_auth_rejection_is_typed_not_empty(self, status):
        """A rejected source fetch must not become fabricated empty content."""

        def handler(method, url, **kw):
            return _FakeResp(status=status, text="denied")

        async with _patch(handler):
            with pytest.raises(GitLabRequestError) as excinfo:
                await GitLabProvider("tok").get_file_content(_PR, "a.py", "feat")

        assert excinfo.value.status_code == status


class TestGetRepoTree:
    @pytest.mark.asyncio
    async def test_lists_blob_paths_only(self):
        def handler(method, url, **kw):
            assert "/repository/tree" in url
            return _FakeResp(
                json_data=[
                    {"type": "blob", "path": "a.py"},
                    {"type": "tree", "path": "pkg"},
                    {"type": "blob", "path": "pkg/b.py"},
                ]
            )

        async with _patch(handler):
            paths = await GitLabProvider("tok").get_repo_tree(_PR, "main")
        assert paths == ["a.py", "pkg/b.py"]


class TestGetFileHistory:
    @pytest.mark.asyncio
    async def test_maps_commits(self):
        def handler(method, url, **kw):
            assert "/repository/commits" in url
            return _FakeResp(
                json_data=[
                    {
                        "short_id": "abc1234",
                        "message": "fix thing\n\ndetails",
                        "author_name": "Alice",
                        "authored_date": "2026-01-01",
                    }
                ]
            )

        async with _patch(handler):
            hist = await GitLabProvider("tok").get_file_history(_PR, ["a.py"], max_per_file=5)
        entry = hist["a.py"][0]
        assert entry.sha == "abc1234"
        assert entry.author == "Alice"
        assert entry.message == "fix thing"

    @pytest.mark.asyncio
    async def test_empty_paths_short_circuits(self):
        async with _patch(lambda *a, **k: _FakeResp(json_data=[])):
            assert await GitLabProvider("tok").get_file_history(_PR, []) == {}


class TestExhaustedFetchSemantics:
    """An exhausted GET is a typed failure, never fabricated absence."""

    @pytest.fixture
    def no_retry_sleeps(self, monkeypatch: pytest.MonkeyPatch):
        async def _sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(gitlab_http, "_sleep", _sleep)

    @pytest.mark.asyncio
    async def test_exhausted_source_get_raises_typed_error(self, no_retry_sleeps):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, headers={"retry-after": "0"})

        provider = GitLabProvider("tok")
        async with _mock_scope(handler):
            with pytest.raises(GitLabRequestError) as excinfo:
                await provider.get_file_content(_PR, "a.py", "feat")

        assert excinfo.value.status_code == 503
        assert excinfo.value.attempts == 5
        assert len(requests) == 5

    @pytest.mark.asyncio
    async def test_missing_file_404_still_returns_empty_sentinel(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        async with _mock_scope(handler):
            assert await GitLabProvider("tok").get_file_content(_PR, "gone.py", "feat") == ""

    @pytest.mark.asyncio
    async def test_exhausted_tree_get_raises_typed_error(self, no_retry_sleeps):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, headers={"retry-after": "0"})

        async with _mock_scope(handler):
            with pytest.raises(GitLabRequestError):
                await GitLabProvider("tok").get_repo_tree(_PR, "main")

        assert len(requests) == 5

    @pytest.mark.asyncio
    async def test_history_keeps_successful_entries_and_names_failed_path(
        self, no_retry_sleeps, caplog
    ):
        requests: list[httpx.Request] = []
        commits = [
            {
                "short_id": "abc12345",
                "message": "fix thing",
                "author_name": "Alice",
                "authored_date": "2026-01-01",
            }
        ]

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.params.get("path") == "bad.py":
                return httpx.Response(503, headers={"retry-after": "0"})
            return httpx.Response(200, json=commits)

        with caplog.at_level(logging.WARNING, logger="mira.providers.gitlab"):
            async with _mock_scope(handler):
                hist = await GitLabProvider("tok").get_file_history(
                    _PR, ["a.py", "bad.py"], max_per_file=5
                )

        assert set(hist) == {"a.py"}
        assert hist["a.py"][0].sha == "abc12345"
        assert "bad.py" in caplog.text
        assert len(requests) == 6


class TestGetRepoPermission:
    @pytest.mark.asyncio
    async def test_looks_up_member_by_username(self):
        def handler(method, url, **kw):
            assert "/members/all" in url
            assert "query=alice" in url
            return _FakeResp(json_data=[{"username": "alice", "access_level": 30}])

        async with _patch(handler):
            permission = await GitLabProvider("tok").get_repo_permission("group", "proj", "alice")
        assert permission == "write"

    @pytest.mark.asyncio
    async def test_looks_up_member_by_user_id(self):
        def handler(method, url, **kw):
            assert "/members/all/42" in url
            return _FakeResp(json_data={"username": "alice", "access_level": 20})

        async with _patch(handler):
            permission = await GitLabProvider("tok").get_repo_permission(
                "group", "proj", "alice", user_id=42
            )
        assert permission == "read"

    @pytest.mark.asyncio
    async def test_missing_member_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(json_data=[])

        async with _patch(handler):
            permission = await GitLabProvider("tok").get_repo_permission("group", "proj", "alice")
        assert permission == "none"

    @pytest.mark.asyncio
    async def test_missing_user_id_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(status=404, json_data={})

        async with _patch(handler):
            permission = await GitLabProvider("tok").get_repo_permission(
                "group", "proj", "alice", user_id=42
            )
        assert permission == "none"


class TestRegistry:
    def test_create_provider_gitlab(self):
        assert isinstance(create_provider("gitlab", "tok"), GitLabProvider)


class TestSharedExecutorRouting:
    """Consumers run inside one runtime scope; retries come from the executor."""

    @pytest.fixture
    def fast_gitlab_retries(self, monkeypatch: pytest.MonkeyPatch):
        async def _sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(gitlab_http, "_sleep", _sleep)

    @staticmethod
    def _provider() -> GitLabProvider:
        return GitLabProvider("tok")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [503, 429])
    async def test_source_get_recovers_and_resends_same_url(self, status):
        requests: list[httpx.Request] = []
        body = "x = 1\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(status, headers={"retry-after": "0"})
            assert request.url.path.endswith("/repository/files/a.py/raw")
            return httpx.Response(200, text=body)

        async with _mock_scope(handler):
            content = await self._provider().get_file_content(_PR, "a.py", "feat")

        assert content == body
        assert len(requests) == 2
        assert requests[0].url == requests[1].url
        assert requests[0].headers["private-token"] == "tok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [503, 429])
    async def test_pagination_recovers_without_duplicate_entries(self, status):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(status, headers={"retry-after": "0"})
            if len(requests) == 2:
                next_url = f"{request.url}&page_token=2"
                return httpx.Response(
                    200,
                    json=[{"id": 1}],
                    headers={"link": f'<{next_url}>; rel="next"'},
                )
            return httpx.Response(200, json=[{"id": 2}])

        provider = self._provider()
        async with _mock_scope(handler):
            items = await provider._paginate(f"{provider._api}/projects/x%2Fy/notes")

        assert [item["id"] for item in items] == [1, 2]
        assert len(requests) == 3
        # The failed first page was retried on the same URL, not re-fetched.
        assert requests[0].url == requests[1].url
        assert "page_token=2" in str(requests[2].url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [503, 429])
    async def test_history_recovers_and_warns_on_failed_paths(self, status, caplog):
        requests: list[httpx.Request] = []
        commits = [
            {
                "short_id": "abc12345",
                "message": "fix thing",
                "author_name": "Alice",
                "authored_date": "2026-01-01",
            }
        ]

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(status, headers={"retry-after": "0"})
            return httpx.Response(200, json=commits)

        with caplog.at_level(logging.WARNING):
            async with _mock_scope(handler):
                hist = await self._provider().get_file_history(_PR, ["a.py"], max_per_file=5)

        assert hist["a.py"][0].sha == "abc12345"
        assert len(requests) == 2
        assert requests[0].url == requests[1].url
        assert dict(requests[0].url.params)["path"] == "a.py"

    @pytest.mark.asyncio
    async def test_history_warns_on_failed_path(self, caplog, fast_gitlab_retries):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, headers={"retry-after": "0"})

        with caplog.at_level(logging.WARNING, logger="mira.providers.gitlab"):
            async with _mock_scope(handler):
                hist = await self._provider().get_file_history(_PR, ["a.py"], max_per_file=5)

        assert hist == {}
        assert len(requests) == 5
        assert "a.py" in caplog.text

    @pytest.mark.asyncio
    async def test_exhaustion_raises_typed_error_after_five_attempts(self):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, headers={"retry-after": "0"})

        provider = self._provider()
        async with _mock_scope(handler):
            with pytest.raises(GitLabRequestError) as excinfo:
                await provider._request("GET", f"{provider._api}/user")

        assert excinfo.value.attempts == 5
        assert excinfo.value.status_code == 503
        assert len(requests) == 5

    @pytest.mark.asyncio
    async def test_404_is_not_retried(self):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(404)

        provider = self._provider()
        async with _mock_scope(handler):
            resp = await provider._request(
                "GET", f"{provider._api}/merge_requests/1", ok=(200, 404)
            )

        assert resp.status_code == 404
        assert len(requests) == 1


class TestTargetSafety:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "next_link",
        [
            '<https://evil.example/api/v4/projects/x%2Fy/notes?page_token=2>; rel="next"',
            '<https://gitlab.com/admin/projects?page_token=2>; rel="next"',
        ],
    )
    async def test_link_target_outside_origin_or_root_is_never_requested(self, next_link):
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json=[{"id": 1}],
                headers={"link": next_link},
            )

        provider = GitLabProvider("tok")
        async with _mock_scope(handler):
            with pytest.raises(GitLabRequestError):
                await provider._paginate(f"{provider._api}/projects/x%2Fy/notes")

        assert len(requests) == 1
        assert requests[0].url.host == "gitlab.com"


class TestDiagnostics:
    @pytest.mark.asyncio
    async def test_logs_exclude_token_query_and_body_secrets(self, caplog):
        secret_token = "gl-super-secret-token"
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, headers={"retry-after": "0"})

        provider = GitLabProvider(secret_token)
        with (
            caplog.at_level(logging.DEBUG, logger="mira.platforms.gitlab.http"),
            caplog.at_level(logging.DEBUG, logger="mira.providers.gitlab"),
        ):
            async with _mock_scope(handler):
                with pytest.raises(GitLabRequestError):
                    await provider._request(
                        "GET",
                        f"{provider._api}/projects/p%2Fq/merge_requests/1?token=query-secret",
                    )
                with pytest.raises(ProviderError):
                    await provider._request(
                        "POST",
                        f"{provider._api}/projects/p%2Fq/notes",
                        data={"body": "body-secret"},
                    )

        text = caplog.text
        assert secret_token not in text
        assert "query-secret" not in text
        assert "body-secret" not in text
        assert len(requests) == 6  # five exhausted GET attempts + one POST
