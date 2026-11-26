"""Tests for the GitLab webhook route + event dispatch."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from mira.config import FilterConfig, MiraConfig
from mira.platforms.gitlab import http as gitlab_http
from mira.platforms.gitlab.auth import GitLabTokenAuth
from mira.platforms.gitlab.http import (
    GitLabHttpRuntime,
    GitLabRequestError,
    gitlab_http_scope,
)
from mira.platforms.server import create_app

GL_SECRET = "gl-secret"
BOT = "mira-bot"


@pytest.fixture
def fast_gitlab_retries(monkeypatch: pytest.MonkeyPatch):
    """Skip executor backoff sleeps while keeping retries and sends real."""

    async def _sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(gitlab_http, "_sleep", _sleep)


class _ManualClock:
    """Deterministic monotonic clock; sleeps are released by the test."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now


async def _next_sleep(
    sleep_calls: asyncio.Queue[tuple[float, asyncio.Event]],
    *,
    what: str,
    timeout: float = 2.0,
) -> tuple[float, asyncio.Event]:
    """Bounded read of the next injected sleep; a regressed gate fails fast."""
    try:
        return await asyncio.wait_for(sleep_calls.get(), timeout=timeout)
    except TimeoutError as exc:
        raise AssertionError(what) from exc


async def _cancel_and_await(tasks: list[asyncio.Task]) -> None:
    """Cancel every tracked task and await completion before runtime teardown."""
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_discovery_gate_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bypass_gate: bool = False,
    tasks: list[asyncio.Task] | None = None,
    wait_timeout: float = 2.0,
) -> tuple[httpx.Response, list[dict[str, object]], list[httpx.AsyncClient]]:
    """Drive discovery plus a sibling provider through one owned runtime.

    ``bypass_gate`` skips the executor's pre-dispatch gate to emulate the
    regression where a throttled sibling never waits; the bounded sleep read
    must then fail instead of hanging. Created tasks are appended to ``tasks``
    when supplied, and are always cancelled and awaited before close.
    """
    from mira.platforms.gitlab import webhook as gw
    from mira.providers.gitlab import GitLabProvider

    clock = _ManualClock()
    sleep_calls: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()

    async def fake_sleep(seconds: float) -> None:
        release = asyncio.Event()
        sleep_calls.put_nowait((seconds, release))
        await release.wait()
        clock.now += seconds

    monkeypatch.setattr(gitlab_http, "_monotonic", clock.time)
    monkeypatch.setattr(gitlab_http, "_sleep", fake_sleep)

    discovery_sends = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal discovery_sends
        if request.url.path.endswith("/projects"):
            discovery_sends += 1
            if discovery_sends == 1:
                return httpx.Response(429, headers={"retry-after": "5"})
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"username": "bot"})

    clients: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    runtime = GitLabHttpRuntime(client_factory=factory)
    if bypass_gate:

        async def _skip_gate(state, deadline, attempts) -> None:
            return None

        monkeypatch.setattr(gitlab_http, "_enter_endpoint", _skip_gate)

    provider = GitLabProvider("tok")
    tracked = tasks if tasks is not None else []
    result: httpx.Response | None = None
    projects: list[dict[str, object]] = []
    async with gitlab_http_scope(runtime):
        try:
            discovery = asyncio.create_task(
                gw.list_gitlab_projects("tok", "https://gitlab.com/api/v4"),
                name="discovery-gate",
            )
            tracked.append(discovery)
            # The executor records the cooldown before entering its retry sleep.
            discovery_delay, release_discovery = await _next_sleep(
                sleep_calls,
                what="discovery never waited on its Retry-After",
                timeout=wait_timeout,
            )
            assert discovery_delay == 5.0

            gate_wait = asyncio.create_task(
                provider._request("GET", f"{provider._api}/user"),
                name="provider-gate",
            )
            tracked.append(gate_wait)
            # The sibling provider request must wait out the shared cooldown.
            provider_delay, release_provider = await _next_sleep(
                sleep_calls,
                what="provider did not wait on the shared gate",
                timeout=wait_timeout,
            )
            assert provider_delay == 5.0

            release_provider.set()
            result = await gate_wait
            release_discovery.set()
            projects = await discovery
        finally:
            await _cancel_and_await(tracked)
            await runtime.aclose()

    assert result is not None
    return result, projects, clients


@pytest.fixture
def gitlab_auth():  # noqa: ANN201
    auth = GitLabTokenAuth("tok")
    # Avoid a live /user call for self-author detection.
    auth.get_bot_identity = AsyncMock(return_value="mira-bot")  # type: ignore[method-assign]
    return auth


@pytest.fixture
def app(gitlab_auth):  # noqa: ANN001, ANN201
    return create_app(
        app_auth=None,
        webhook_secret=None,
        bot_name=BOT,
        gitlab_auth=gitlab_auth,
        gitlab_webhook_secret=GL_SECRET,
    )


@pytest.fixture
async def client(app) -> AsyncClient:  # noqa: ANN001
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _mr_payload(action="open", oldrev=None):
    attrs = {
        "iid": 7,
        "action": action,
        "source_branch": "f",
        "target_branch": "main",
        "url": "https://gitlab.com/g/p/-/merge_requests/7",
    }
    if oldrev:
        attrs["oldrev"] = oldrev
    return {
        "object_kind": "merge_request",
        "object_attributes": attrs,
        "project": {
            "path_with_namespace": "g/p",
            "web_url": "https://gitlab.com/g/p",
            "default_branch": "main",
            "visibility": "private",
        },
        "user": {"username": "alice"},
    }


@pytest.mark.asyncio
async def test_rejects_bad_token(client):
    resp = await client.post(
        "/gitlab/webhook",
        content=json.dumps(_mr_payload()),
        headers={"X-Gitlab-Token": "wrong", "X-Gitlab-Event": "Merge Request Hook"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_mr_open_triggers_review(client):
    with patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(_mr_payload(action="open")),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_mr_update_without_newcommits_ignored(client):
    with patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(_mr_payload(action="update")),  # no oldrev = no new commits
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_self_authored_event_ignored(client, gitlab_auth):
    payload = _mr_payload(action="open")
    payload["user"]["username"] = "mira-bot"  # the bot itself
    with patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


def _note_payload(note="@mira-bot review"):
    return {
        "object_kind": "note",
        "object_attributes": {"note": note, "noteable_type": "MergeRequest"},
        "merge_request": {"iid": 7, "url": "https://gitlab.com/g/p/-/merge_requests/7"},
        "project": {"path_with_namespace": "g/p", "web_url": "https://gitlab.com/g/p"},
        "user": {"username": "alice"},
    }


@pytest.mark.asyncio
async def test_note_mention_triggers_command(client):
    with patch("mira.platforms.gitlab.webhook.handle_gitlab_note", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(_note_payload()),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Note Hook"},
        )
    assert resp.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_note_without_mention_ignored(client):
    with patch("mira.platforms.gitlab.webhook.handle_gitlab_note", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(_note_payload(note="looks good")),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Note Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_mr_merge_triggers_learning(client):
    with patch("mira.platforms.gitlab.webhook.handle_gitlab_merge", new=AsyncMock()) as h:
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(_mr_payload(action="merge")),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_inline_mention_routes_to_thread_reply(gitlab_auth):
    """A free-form @-mention on a diff note → LLM intent classification."""
    from mira.platforms.gitlab import webhook as gw

    payload = {
        "object_attributes": {
            "note": "@mira-bot are you sure?",
            "noteable_type": "MergeRequest",
            "id": 99,
            "discussion_id": "abc123",
            "position": {"new_path": "a.py", "new_line": 4},
        },
        "merge_request": {"iid": 7, "url": "https://gitlab.com/g/p/-/merge_requests/7"},
        "project": {"path_with_namespace": "g/p", "web_url": "https://gitlab.com/g/p"},
        "user": {"username": "alice"},
    }
    prov = AsyncMock()
    prov.get_pr_info = AsyncMock(return_value=object())
    with (
        patch("mira.platforms.gitlab.webhook.create_provider", return_value=prov),
        patch("mira.platforms.handlers.run_thread_reply", new=AsyncMock()) as rtr,
        patch("mira.platforms.handlers.run_pr_command", new=AsyncMock()) as rpc,
    ):
        await gw.handle_gitlab_note(payload, gitlab_auth, "mira-bot")
    rtr.assert_awaited_once()
    rpc.assert_not_called()


@pytest.mark.asyncio
async def test_paused_label_skips_review(gitlab_auth):
    from mira.platforms.gitlab import webhook as gw

    payload = _mr_payload(action="open")
    payload["labels"] = [{"title": "mira-paused"}]
    with patch("mira.platforms.handlers.run_pr_review", new=AsyncMock()) as rpr:
        await gw.handle_merge_request(payload, gitlab_auth, "mira-bot")
    rpr.assert_not_called()


@pytest.mark.asyncio
async def test_ignore_in_description_skips_review(gitlab_auth):
    from mira.platforms.gitlab import webhook as gw

    payload = _mr_payload(action="open")
    payload["object_attributes"]["description"] = "wip @mira-bot ignore for now"
    with patch("mira.platforms.handlers.run_pr_review", new=AsyncMock()) as rpr:
        await gw.handle_merge_request(payload, gitlab_auth, "mira-bot")
    rpr.assert_not_called()


@pytest.mark.asyncio
async def test_unpaused_mr_runs_review(gitlab_auth):
    from mira.platforms.gitlab import webhook as gw

    payload = _mr_payload(action="open")
    with (
        patch("mira.platforms.index_handlers._get_app_db"),
        patch("mira.platforms.gitlab.webhook.create_provider", return_value=AsyncMock()),
        patch("mira.platforms.handlers.run_pr_review", new=AsyncMock()) as rpr,
    ):
        await gw.handle_merge_request(payload, gitlab_auth, "mira-bot")
    rpr.assert_awaited_once()


@pytest.mark.asyncio
async def test_github_route_404_when_not_configured(client):
    # GitLab-only deployment: the GitHub webhook route is disabled.
    resp = await client.post("/github/webhook", content="{}", headers={"X-GitHub-Event": "ping"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_mr_open_target_branch_filtered(client):
    payload = _mr_payload(action="open")
    payload["object_attributes"]["target_branch"] = "develop"
    cfg = MiraConfig(filter=FilterConfig(allowed_base_branches=["main"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_mr_open_target_branch_allowed(client):
    payload = _mr_payload(action="open")
    payload["object_attributes"]["target_branch"] = "main"
    cfg = MiraConfig(filter=FilterConfig(allowed_base_branches=["main"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_mr_note_mention_ignores_branch_policy(client):
    # Manual commands are never gated by the branch policy.
    payload = _note_payload("@mira-bot review")
    cfg = MiraConfig(filter=FilterConfig(allowed_base_branches=["main"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_gitlab_note", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Note Hook"},
        )
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_mr_open_blocked_author_filtered(client):
    payload = _mr_payload(action="open")
    payload["user"]["username"] = "dependabot"
    cfg = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_mr_open_allowed_author_not_filtered(client):
    payload = _mr_payload(action="open")
    payload["user"]["username"] = "alice"
    cfg = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_mr_open_allowlist_filters_off_list(client):
    payload = _mr_payload(action="open")
    payload["user"]["username"] = "bob"
    cfg = MiraConfig(filter=FilterConfig(allowed_authors=["alice"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_merge_request", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"},
        )
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_note_review_ignores_author_filter(client):
    """Commands use the command policy, not the auto-review author filter."""
    payload = _note_payload(note="@mira-bot review")
    payload["user"]["username"] = "dependabot"
    cfg = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_gitlab_note", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Note Hook"},
        )
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_note_non_review_ignores_author_filter(client):
    """Free-form command dispatch is separate from the auto-review author filter."""
    payload = _note_payload(note="@mira-bot pause")
    payload["user"]["username"] = "alice"
    cfg = MiraConfig(filter=FilterConfig(blocked_authors=["alice"]))
    with (
        patch("mira.platforms.gitlab.webhook.load_config", return_value=cfg),
        patch("mira.platforms.gitlab.webhook.handle_gitlab_note", new=AsyncMock()) as h,
    ):
        resp = await client.post(
            "/gitlab/webhook",
            content=json.dumps(payload),
            headers={"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Note Hook"},
        )
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


def _gitlab_handler(projects, *, requests=None):
    """MockTransport handler honouring ``active=true`` like GitLab."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        url = str(request.url)
        if "active=true" in url:
            # active=true = not archived and not marked for deletion.
            visible = [
                p for p in projects if not p.get("archived") and not p.get("pending_deletion")
            ]
        else:
            visible = projects
        return httpx.Response(200, json=visible)

    return handler


@asynccontextmanager
async def _patch_gitlab_http(projects, *, requests=None):
    runtime = GitLabHttpRuntime(
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(_gitlab_handler(projects, requests=requests))
        )
    )
    async with gitlab_http_scope(runtime):
        try:
            yield runtime
        finally:
            await runtime.aclose()


@pytest.fixture
def gitlab_project_list():
    return [
        {"path_with_namespace": "netlife/wfp", "visibility": "private"},
        {"path_with_namespace": "norel/douglas", "visibility": "private", "archived": True},
        {
            "path_with_namespace": "netlife/opencode.netlife.pl-deletion_scheduled-88322484",
            "visibility": "private",
            "pending_deletion": True,
        },
    ]


@pytest.mark.asyncio
async def test_list_gitlab_projects_filters_inactive_server_side(gitlab_project_list):
    from mira.platforms.gitlab import webhook as gw

    requests: list[httpx.Request] = []
    async with _patch_gitlab_http(gitlab_project_list, requests=requests):
        projects = await gw.list_gitlab_projects("tok", "https://gitlab.com/api/v4")

    assert "active=true" in str(requests[0].url)
    assert "include_pending_delete" not in str(requests[0].url)
    assert [p["path_with_namespace"] for p in projects] == ["netlife/wfp"]


@pytest.mark.asyncio
async def test_backfill_skips_archived_and_pending_deletion(gitlab_auth, gitlab_project_list):
    from mira.platforms.gitlab import webhook as gw

    db = MagicMock()
    async with _patch_gitlab_http(gitlab_project_list):
        with patch("mira.platforms.index_handlers._get_app_db", return_value=db):
            count = await gw.backfill_gitlab_projects(gitlab_auth)

    # The sync endpoint reports this count; archived and pending-deletion repos
    # must not be registered or included.
    assert count == 1
    db.register_repo.assert_called_once_with("netlife", "wfp", platform="gitlab")
    db.set_repo_visibility.assert_called_once_with("netlife", "wfp", True, platform="gitlab")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 429])
async def test_discovery_recovers_from_retryable_status(gitlab_project_list, status):
    from mira.platforms.gitlab import webhook as gw

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, headers={"retry-after": "0"})
        return httpx.Response(200, json=gitlab_project_list)

    runtime = GitLabHttpRuntime(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    async with gitlab_http_scope(runtime):
        try:
            projects = await gw.list_gitlab_projects("tok", "https://gitlab.com/api/v4")
        finally:
            await runtime.aclose()

    assert [p["path_with_namespace"] for p in projects] == [
        "netlife/wfp",
        "norel/douglas",
        "netlife/opencode.netlife.pl-deletion_scheduled-88322484",
    ]
    assert len(requests) == 2
    assert requests[0].url == requests[1].url


@pytest.mark.asyncio
async def test_discovery_exhaustion_raises_not_partial_prefix(fast_gitlab_retries):
    """A successful page 1 must not be returned when page 2 exhausts."""
    from mira.platforms.gitlab import webhook as gw

    requests: list[httpx.Request] = []
    statuses: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page") == "2":
            statuses.append(503)
            return httpx.Response(503, headers={"retry-after": "0"})
        statuses.append(200)
        next_url = f"{request.url}&page=2"
        return httpx.Response(
            200,
            json=[{"path_with_namespace": "netlife/wfp", "visibility": "private"}],
            headers={"link": f'<{next_url}>; rel="next"'},
        )

    runtime = GitLabHttpRuntime(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    async with gitlab_http_scope(runtime):
        try:
            with pytest.raises(GitLabRequestError) as excinfo:
                await gw.list_gitlab_projects("tok", "https://gitlab.com/api/v4")
        finally:
            await runtime.aclose()

    assert excinfo.value.attempts == 5
    # Page 1 succeeds, then page 2 is retried five times without a prefix result.
    assert statuses == [200, 503, 503, 503, 503, 503]
    assert len(requests) == 6
    assert all(request.url.params.get("page") == "2" for request in requests[1:])
    assert len({str(request.url) for request in requests[1:]}) == 1


@pytest.mark.asyncio
async def test_discovery_429_gates_concurrent_provider_request(monkeypatch: pytest.MonkeyPatch):
    """A discovery throttle is shared with provider calls on the same token.

    A manual monotonic clock plus per-sleep release events make the gate wait
    deterministic: no real sleeping and no wall-clock assertions.
    """
    result, projects, clients = await _run_discovery_gate_scenario(monkeypatch)
    assert result.json()["username"] == "bot"
    assert projects == []
    assert len(clients) == 1


@pytest.mark.asyncio
async def test_missing_shared_gate_fails_fast_without_pending_tasks(
    monkeypatch: pytest.MonkeyPatch,
):
    """Negative control: a sibling that skips the gate must fail the bounded
    wait promptly and leave no pending tasks after cleanup."""
    tasks: list[asyncio.Task] = []
    with pytest.raises(AssertionError, match="provider did not wait"):
        await _run_discovery_gate_scenario(
            monkeypatch, bypass_gate=True, tasks=tasks, wait_timeout=1.0
        )
    assert tasks
    assert all(task.done() for task in tasks)
