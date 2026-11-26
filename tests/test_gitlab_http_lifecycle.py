"""Task 4: server/CLI ownership of the shared GitLab HTTP runtime.

These tests enter ``app.router.lifespan_context(app)`` explicitly: HTTPX's
ASGITransport does not run FastAPI's lifespan automatically.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.background import BackgroundTasks
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mira.dashboard import api as dashboard_api
from mira.dashboard import auth as dashboard_auth
from mira.dashboard.routers import repos as repos_routes
from mira.platforms.gitlab import http as gitlab_http
from mira.platforms.gitlab.auth import GitLabTokenAuth
from mira.platforms.gitlab.http import (
    _RUNTIME_CONTEXT,
    GitLabHttpRuntime,
    request_gitlab,
)
from mira.platforms.server import create_app

GL_SECRET = "gl-secret"
BOT = "mira-bot"
TOKEN = "tok"
API = "https://gitlab.example.com/api/v4"


def _make_auth(token: str = TOKEN) -> GitLabTokenAuth:
    auth = GitLabTokenAuth(token)
    auth.get_token = AsyncMock(return_value=token)  # type: ignore[method-assign]
    auth.get_bot_identity = AsyncMock(return_value=BOT)  # type: ignore[method-assign]
    return auth


async def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


class ClientRecorder:
    """Factory recording pooled clients, their loop and close ordering."""

    def __init__(
        self,
        handler=_ok_handler,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.handler = handler
        self.events = events if events is not None else []
        self.created: list[httpx.AsyncClient] = []
        self.loops: list[asyncio.AbstractEventLoop | None] = []

    def __call__(self) -> httpx.AsyncClient:
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self.loops.append(loop)
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        original_close = client.aclose

        async def tracked_close() -> None:
            await original_close()
            self.events.append("client-closed")

        client.aclose = tracked_close  # type: ignore[method-assign]
        self.created.append(client)
        return client


class ManualClock:
    """Clock whose sleeps block until the test releases them."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self._events: list[asyncio.Event] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        event = asyncio.Event()
        self._events.append(event)
        self.sleeps.append(seconds)
        await event.wait()
        self.now += seconds

    def release_all(self) -> None:
        for event in self._events:
            event.set()


@pytest.fixture
def manual_clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    clock = ManualClock()
    monkeypatch.setattr(gitlab_http, "_monotonic", clock.time)
    monkeypatch.setattr(gitlab_http, "_sleep", clock.sleep)
    return clock


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(gitlab_http, "_sleep", _sleep)


async def wait_until(predicate: Any, *, what: str) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


def _install_probe_routes(app) -> None:
    """Insert a probe route ahead of any catch-all SPA fallback."""

    async def probe(request: Request) -> JSONResponse:
        body = await request.body()
        suffix = body.decode() or "/ok"
        try:
            response = await request_gitlab(TOKEN, API, "GET", f"{API}{suffix}")
        except gitlab_http.GitLabRequestError as exc:
            return JSONResponse({"error": exc.reason}, status_code=502)
        return JSONResponse({"status": response.status_code})

    app.router.routes.insert(0, Route("/probe/job", probe, methods=["POST"]))


async def _noop_discovery(auth: Any) -> int:
    return 0


def _build_app(auth: GitLabTokenAuth | None = None) -> Any:
    return create_app(
        app_auth=None,
        webhook_secret=None,
        bot_name=BOT,
        gitlab_auth=auth if auth is not None else _make_auth(),
        gitlab_webhook_secret=GL_SECRET,
    )


@asynccontextmanager
async def running_app(app: Any):
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


# ---------------------------------------------------------------------------
# Shared ownership across requests, background work and dashboard jobs
# ---------------------------------------------------------------------------


async def test_requests_background_and_dashboard_jobs_share_one_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)

    webhook_entered = 0
    both_entered = asyncio.Event()
    webhook_bg_done = asyncio.Event()
    dashboard_done = asyncio.Event()
    runtimes: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal webhook_entered
        path = request.url.path
        if path.endswith("/webhook-request"):
            webhook_entered += 1
            if webhook_entered >= 2:
                both_entered.set()
            await both_entered.wait()
            return httpx.Response(200)
        return httpx.Response(200)

    async def discovery(auth: Any) -> int:
        runtimes["discovery"] = _RUNTIME_CONTEXT.get()
        await request_gitlab(TOKEN, API, "GET", f"{API}/discovery")
        return 0

    async def webhook_job() -> None:
        runtimes["webhook-background"] = _RUNTIME_CONTEXT.get()
        await request_gitlab(TOKEN, API, "GET", f"{API}/webhook-background")
        webhook_bg_done.set()

    async def dispatch(event, payload, auth, bot_name, background_tasks) -> str:
        runtimes["dispatch"] = _RUNTIME_CONTEXT.get()
        await request_gitlab(TOKEN, API, "GET", f"{API}/webhook-request")
        background_tasks.add_task(webhook_job)
        return "processing"

    async def index_job() -> None:
        runtimes["dashboard-job"] = _RUNTIME_CONTEXT.get()
        await request_gitlab(TOKEN, API, "GET", f"{API}/index-job")
        dashboard_done.set()

    async def start_index(request: Request) -> JSONResponse:
        asyncio.create_task(index_job())
        return JSONResponse({"status": "indexing"})

    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", discovery)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.dispatch_gitlab_event", dispatch)

    app = _build_app()
    app.router.routes.insert(0, Route("/probe/index-job", start_index, methods=["POST"]))

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        assert isinstance(runtime, GitLabHttpRuntime)

        headers = {"X-Gitlab-Token": GL_SECRET, "X-Gitlab-Event": "Merge Request Hook"}
        payload = {"object_kind": "merge_request", "user": {"username": "alice"}}
        first = asyncio.create_task(client.post("/gitlab/webhook", json=payload, headers=headers))
        second = asyncio.create_task(client.post("/gitlab/webhook", json=payload, headers=headers))
        responses = await asyncio.gather(first, second)
        assert [response.json()["status"] for response in responses] == [
            "processing",
            "processing",
        ]
        await asyncio.wait_for(webhook_bg_done.wait(), 5)

        index_response = await client.post("/probe/index-job")
        assert index_response.status_code == 200
        await asyncio.wait_for(dashboard_done.wait(), 5)

        expected = {
            "discovery",
            "dispatch",
            "webhook-background",
            "dashboard-job",
        }
        assert expected <= set(runtimes)
        for source in expected:
            assert runtimes[source] is runtime, source
        assert len(recorder.created) == 1
        assert recorder.created[0].is_closed is False
        assert all(loop is not None for loop in recorder.loops)

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True


async def test_request_completion_does_not_close_lifespan_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)
    app = _build_app()
    _install_probe_routes(app)

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        response = await client.post("/probe/job", content="/ok")
        assert response.status_code == 200
        assert len(recorder.created) == 1
        assert recorder.created[0].is_closed is False
        assert runtime._closed is False

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True


async def test_request_background_task_is_drained_before_lifespan_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    started = asyncio.Event()
    release = asyncio.Event()
    done = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/slow-background"):
            started.set()
            await release.wait()
        return httpx.Response(200)

    recorder.handler = handler

    async def background_job() -> None:
        await request_gitlab(TOKEN, API, "GET", f"{API}/slow-background")
        done.set()

    async def route(request: Request) -> JSONResponse:
        background = BackgroundTasks()
        background.add_task(background_job)
        response = JSONResponse({"ok": True})
        response.background = background
        return response

    app = _build_app()
    app.router.routes.insert(0, Route("/probe/background", route, methods=["POST"]))

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        pending = asyncio.create_task(client.post("/probe/background"))
        await started.wait()
        assert pending.done() is False
        release.set()
        response = await pending
        assert response.status_code == 200
        await asyncio.wait_for(done.wait(), 5)
        assert runtime._closed is False
        assert recorder.created[0].is_closed is False

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True


# ---------------------------------------------------------------------------
# Shutdown ordering and cancellation-safe discovery
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_and_awaits_discovery_before_closing_clients(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _no_sleep(monkeypatch)
    events: list[str] = []
    recorder = ClientRecorder(events=events)
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)

    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/discovery-slow"):
            started.set()
            await asyncio.Event().wait()
        return httpx.Response(200)

    recorder.handler = handler

    async def discovery(auth: Any) -> int:
        try:
            await request_gitlab(TOKEN, API, "GET", f"{API}/discovery-slow")
        except asyncio.CancelledError:
            events.append("discovery-cancelled")
            raise
        return 0

    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", discovery)
    app = _build_app()

    with caplog.at_level(logging.ERROR):
        async with app.router.lifespan_context(app):
            await started.wait()
            runtime = app.state.gitlab_http_runtime
            assert len(recorder.created) == 1
            assert recorder.created[0].is_closed is False
        await asyncio.sleep(0)

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True
    assert events == ["discovery-cancelled", "client-closed"]
    assert "Exception in callback" not in caplog.text


# ---------------------------------------------------------------------------
# Failure and cancellation safety
# ---------------------------------------------------------------------------


async def test_sibling_request_waits_for_cooldown_shared_across_requests(
    monkeypatch: pytest.MonkeyPatch,
    manual_clock: ManualClock,
) -> None:
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    limited_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal limited_calls
        if request.url.path.endswith("/limited"):
            limited_calls += 1
            if limited_calls == 1:
                return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200)

    recorder.handler = handler
    app = _build_app()
    _install_probe_routes(app)

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        first = asyncio.create_task(client.post("/probe/job", content="/limited"))
        await wait_until(lambda: len(manual_clock.sleeps) == 1, what="retry sleep")
        second = asyncio.create_task(client.post("/probe/job", content="/ok"))
        await wait_until(lambda: len(manual_clock.sleeps) == 2, what="sibling gate wait")
        assert manual_clock.sleeps == [5.0, 5.0]
        manual_clock.release_all()
        responses = await asyncio.gather(first, second)
        assert [response.status_code for response in responses] == [200, 200]
        assert len(recorder.created) == 1
        assert recorder.created[0].is_closed is False
        assert runtime._closed is False

    assert recorder.created[0].is_closed is True


async def test_failed_request_keeps_runtime_and_client_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/fail"):
            return httpx.Response(500)
        return httpx.Response(200)

    recorder.handler = handler
    app = _build_app()
    _install_probe_routes(app)

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        failed = await client.post("/probe/job", content="/fail")
        assert failed.status_code == 502
        ok = await client.post("/probe/job", content="/ok")
        assert ok.status_code == 200
        assert len(recorder.created) == 1
        assert recorder.created[0].is_closed is False
        assert runtime._closed is False

    assert recorder.created[0].is_closed is True


async def test_cancelled_request_releases_permit_and_runtime_stays_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/block"):
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200)

    recorder.handler = handler
    app = _build_app()
    _install_probe_routes(app)

    async with running_app(app) as client:
        runtime = app.state.gitlab_http_runtime
        pending = asyncio.create_task(client.post("/probe/job", content="/block"))
        await entered.wait()
        state = next(iter(runtime._states.values()))
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        ok = await client.post("/probe/job", content="/ok")
        assert ok.status_code == 200
        assert state.semaphore._value == gitlab_http._MAX_ACTIVE_REQUESTS
        assert len(recorder.created) == 1
        assert recorder.created[0].is_closed is False
        assert runtime._closed is False

    assert recorder.created[0].is_closed is True


async def test_dashboard_gitlab_index_job_is_drained_before_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``POST /api/repos/{owner}/{repo}/index`` spawn boundary must be
    tracked, and its terminal cleanup must finish before pooled clients close.
    """
    _no_sleep(monkeypatch)
    events: list[str] = []
    recorder = ClientRecorder(events=events)
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/index-job"):
            entered.set()
            await release.wait()
        return httpx.Response(200)

    recorder.handler = handler

    # Exercise the actual dashboard route/spawn boundary.
    repo_path = "/api/repos/g/lifecycle-index-job/index"
    monkeypatch.setattr(dashboard_auth, "_PUBLIC_PATHS", {repo_path})
    monkeypatch.setattr(repos_routes, "_require_admin", lambda request: None)
    monkeypatch.setenv("MIRA_GITLAB_TOKEN", TOKEN)
    record = SimpleNamespace(platform="gitlab")
    monkeypatch.setattr(
        dashboard_api._app_db, "get_repo_any_platform", lambda owner, repo: [record]
    )
    monkeypatch.setattr(dashboard_api._app_db, "set_repo_status", MagicMock())
    monkeypatch.setattr("mira.config.load_config", MagicMock())
    monkeypatch.setattr("mira.llm.create_llm", MagicMock())
    monkeypatch.setattr("mira.dashboard.models_config.llm_config_for", MagicMock())

    store = MagicMock()
    store.claim_index_run.return_value = object()
    store.close.side_effect = lambda: events.append("store-closed")
    monkeypatch.setattr(repos_routes, "IndexStore", MagicMock(open=lambda *args, **kwargs: store))

    async def fake_index_repo(**kwargs: Any) -> int:
        try:
            await request_gitlab(TOKEN, API, "GET", f"{API}/index-job")
            return 1
        finally:
            events.append("job-terminal")

    monkeypatch.setattr("mira.index.indexer.index_repo", fake_index_repo)

    app = _build_app()
    try:
        async with running_app(app) as client:
            runtime = app.state.gitlab_http_runtime
            response = await client.post(repo_path)
            assert response.status_code == 200
            assert response.json()["status"] == "indexing"
            await entered.wait()
            assert len(recorder.created) == 1
            assert recorder.created[0].is_closed is False
            assert runtime._closed is False
    finally:
        release.set()

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True
    assert events[-3:] == ["job-terminal", "store-closed", "client-closed"]


def _patch_trigger_index_route(
    monkeypatch: pytest.MonkeyPatch, *, route_path: str, target: str, events: list[str]
) -> None:
    """Patch the real dashboard ``trigger_index`` spawn boundary for one job."""
    monkeypatch.setattr(dashboard_auth, "_PUBLIC_PATHS", {route_path})
    monkeypatch.setattr(repos_routes, "_require_admin", lambda request: None)
    monkeypatch.setenv("MIRA_GITLAB_TOKEN", TOKEN)
    monkeypatch.setattr(
        dashboard_api._app_db,
        "get_repo_any_platform",
        lambda owner, repo: [SimpleNamespace(platform="gitlab")],
    )
    monkeypatch.setattr(dashboard_api._app_db, "set_repo_status", MagicMock())
    monkeypatch.setattr("mira.config.load_config", MagicMock())
    monkeypatch.setattr("mira.llm.create_llm", MagicMock())
    monkeypatch.setattr("mira.dashboard.models_config.llm_config_for", MagicMock())

    store = MagicMock()
    store.claim_index_run.return_value = object()
    store.close.side_effect = lambda: events.append("store-closed")
    monkeypatch.setattr(repos_routes, "IndexStore", MagicMock(open=lambda *args, **kwargs: store))

    async def fake_index_repo(**kwargs: Any) -> int:
        try:
            await request_gitlab(TOKEN, API, "GET", f"{API}{target}")
            return 1
        finally:
            events.append("job-terminal")

    monkeypatch.setattr("mira.index.indexer.index_repo", fake_index_repo)


async def test_other_app_shutdown_does_not_drain_owners_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing app B must not cancel app A's tracked indexing job."""
    _no_sleep(monkeypatch)
    events: list[str] = []
    recorder = ClientRecorder(events=events)
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/isolation-job"):
            entered.set()
            await release.wait()
        return httpx.Response(200)

    recorder.handler = handler
    route_path = "/api/repos/g/lifecycle-isolation/index"
    _patch_trigger_index_route(
        monkeypatch, route_path=route_path, target="/isolation-job", events=events
    )

    app_one = _build_app()
    app_two = _build_app()
    try:
        async with running_app(app_one) as client_one:
            runtime_one = app_one.state.gitlab_http_runtime
            response = await client_one.post(route_path)
            assert response.status_code == 200
            await entered.wait()
            async with running_app(app_two):
                pass
            # App B is gone; app A's job must still own its work and client.
            assert "job-terminal" not in events
            assert "store-closed" not in events
            assert runtime_one._closed is False
            assert recorder.created[0].is_closed is False
    finally:
        release.set()

    assert runtime_one._closed is True
    assert events[-3:] == ["job-terminal", "store-closed", "client-closed"]


async def test_non_gitlab_setup_body_tracks_pending_gitlab_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GitHub-only setup request must still track GitLab work it selects."""
    _no_sleep(monkeypatch)
    events: list[str] = []
    recorder = ClientRecorder(events=events)
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/setup-job"):
            entered.set()
            await release.wait()
        return httpx.Response(200)

    recorder.handler = handler
    route_path = "/api/setup/complete"
    monkeypatch.setattr(dashboard_auth, "_PUBLIC_PATHS", {route_path})
    monkeypatch.setattr("mira.dashboard.routers.admin._require_admin", lambda request: None)
    monkeypatch.setenv("MIRA_GITLAB_TOKEN", TOKEN)

    # Eligible GitLab repo already pending in the database, even though the
    # setup request body only mentions a GitHub repo.
    pending = SimpleNamespace(
        owner="g", repo="pending-proj", platform="gitlab", index_mode="full", status="indexing"
    )
    monkeypatch.setattr(dashboard_api._app_db, "list_repos", lambda: [pending])
    monkeypatch.setattr(dashboard_api._app_db, "set_repo_index_mode", MagicMock())
    monkeypatch.setattr(dashboard_api._app_db, "set_repo_status", MagicMock())
    monkeypatch.setattr(dashboard_api._app_db, "mark_setup_complete", MagicMock())
    monkeypatch.setattr("mira.config.load_config", MagicMock())
    monkeypatch.setattr("mira.llm.create_llm", MagicMock())
    monkeypatch.setattr("mira.dashboard.models_config.llm_config_for", MagicMock())

    store = MagicMock()
    store.claim_index_run.return_value = object()
    store.close.side_effect = lambda: events.append("store-closed")
    monkeypatch.setattr(dashboard_api, "IndexStore", MagicMock(open=lambda *args, **kwargs: store))

    async def fake_index_repo(**kwargs: Any) -> int:
        try:
            await request_gitlab(TOKEN, API, "GET", f"{API}/setup-job")
            return 1
        finally:
            events.append("job-terminal")

    monkeypatch.setattr("mira.index.indexer.index_repo", fake_index_repo)

    app = _build_app()
    try:
        async with running_app(app) as client:
            runtime = app.state.gitlab_http_runtime
            response = await client.post(
                route_path,
                json={
                    "repos": [
                        {"owner": "acme", "repo": "web", "platform": "github", "enabled": True}
                    ],
                    "index_mode": "full",
                },
            )
            assert response.status_code == 200
            await entered.wait()
            assert len(recorder.created) == 1
            assert recorder.created[0].is_closed is False
            assert runtime._closed is False
    finally:
        release.set()

    assert runtime._closed is True
    assert recorder.created[0].is_closed is True
    assert events[-3:] == ["job-terminal", "store-closed", "client-closed"]


# ---------------------------------------------------------------------------
# Per-app isolation
# ---------------------------------------------------------------------------


async def test_two_apps_do_not_share_runtime_or_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    recorder = ClientRecorder()
    monkeypatch.setattr(gitlab_http, "_default_client_factory", recorder)
    monkeypatch.setattr("mira.platforms.gitlab.webhook.backfill_gitlab_projects", _noop_discovery)

    app_one = _build_app(_make_auth())
    app_two = _build_app(_make_auth())
    _install_probe_routes(app_one)
    _install_probe_routes(app_two)

    runtime_one = runtime_two = None
    async with running_app(app_one) as client_one:
        runtime_one = app_one.state.gitlab_http_runtime
        assert (await client_one.post("/probe/job", content="/ok")).status_code == 200
        async with running_app(app_two) as client_two:
            runtime_two = app_two.state.gitlab_http_runtime
            assert (await client_two.post("/probe/job", content="/ok")).status_code == 200
            assert runtime_one is not runtime_two
            assert len(recorder.created) == 2
            assert recorder.created[0] is not recorder.created[1]
            assert all(not client.is_closed for client in recorder.created)
            assert _RUNTIME_CONTEXT.get() is None
        assert runtime_two._closed is True
        assert recorder.created[1].is_closed is True
        assert runtime_one._closed is False
        assert recorder.created[0].is_closed is False

    assert runtime_one._closed is True
    assert recorder.created[0].is_closed is True
    assert _RUNTIME_CONTEXT.get() is None
