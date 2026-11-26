"""Task 6: real loopback evidence for shared GitLab HTTP pooling.

Mock transports prove retry and throttle policy but cannot prove TCP connection
reuse. These tests drive a real HTTP/1.1 keep-alive server on loopback through
a proxy-free injected HTTPX client and count accepted TCP connections.

``MIRA_GITLAB_HTTP_POOL_LEGACY_CONTROL=1`` reruns the reuse assertion through an
isolated client-per-call harness that mirrors the pre-change behaviour. It
exists only to capture the historical RED evidence; normal runs exercise the
shared runtime.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from contextlib import asynccontextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from mira.platforms.gitlab.http import (
    GitLabHttpRuntime,
    gitlab_http_scope,
    request_gitlab,
)

LEGACY_CONTROL = os.environ.get("MIRA_GITLAB_HTTP_POOL_LEGACY_CONTROL") == "1"


class _KeepAliveHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 handler with explicit Content-Length for keep-alive."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.server.gate is not None:  # type: ignore[attr-defined]
            self.server.gate.wait(timeout=5.0)  # type: ignore[attr-defined]
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Silence request logging; the counter is the observation."""


class _LoopbackServer(ThreadingHTTPServer):
    """Threading HTTP server counting every accepted TCP connection."""

    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(address, handler)
        self.url = ""
        self.connection_count = 0
        self.connections: list[socket.socket] = []
        self.gate: threading.Event | None = None
        self.gate_target = 0
        self._lock = threading.Lock()

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        with self._lock:
            self.connection_count += 1
            self.connections.append(request)
            if self.gate is not None and self.connection_count >= self.gate_target:
                self.gate.set()
        super().process_request(request, client_address)

    def handle_error(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        """Teardown closes tracked sockets while handler threads are blocked."""


@asynccontextmanager
async def loopback_server(*, gate_target: int = 0):
    """Start a loopback keep-alive server and tear it down deterministically."""
    server = _LoopbackServer(("127.0.0.1", 0), _KeepAliveHandler)
    host, port = server.server_address[:2]
    server.url = f"http://{host}:{port}"
    if gate_target:
        server.gate = threading.Event()
        server.gate_target = gate_target
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        for sock in server.connections:
            with suppress(OSError):
                sock.close()
        thread.join(timeout=5)


def _loopback_client(base_url: str) -> httpx.AsyncClient:
    """Proxy-free client configured with the production pool/timeouts."""
    return httpx.AsyncClient(
        base_url=base_url,
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=5.0),
        limits=httpx.Limits(
            max_connections=8,
            max_keepalive_connections=8,
            keepalive_expiry=30.0,
        ),
    )


class _SharedPath:
    """Two logical jobs resolved through one owned, bound runtime."""

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.runtime = GitLabHttpRuntime(client_factory=lambda: _loopback_client(api_url))
        self._scope = gitlab_http_scope(self.runtime)

    async def __aenter__(self) -> _SharedPath:
        await self._scope.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._scope.__aexit__(*exc_info)
        await self.runtime.aclose()

    async def get(self, path: str) -> httpx.Response:
        return await request_gitlab("token", self.api_url, "GET", path)

    @property
    def pooled_client(self) -> httpx.AsyncClient:
        state = next(iter(self.runtime._states.values()))
        client = state.client
        assert client is not None
        return client


class _LegacyClientPerCall:
    """Negative control mirroring the pre-change client-per-call path."""

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    async def __aenter__(self) -> _LegacyClientPerCall:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get(self, path: str) -> httpx.Response:
        client = _loopback_client(self.api_url)
        try:
            request = client.build_request(
                "GET",
                f"{self.api_url}{path}",
                headers={"PRIVATE-TOKEN": "token"},
            )
            return await client.send(request, follow_redirects=False)
        finally:
            await client.aclose()


def _path_for(api_url: str) -> _SharedPath | _LegacyClientPerCall:
    """Select the shared runtime, or the isolated legacy control under RED."""
    if LEGACY_CONTROL:
        return _LegacyClientPerCall(api_url)
    return _SharedPath(api_url)


@pytest.mark.asyncio
async def test_sequential_cross_job_requests_share_one_tcp_connection() -> None:
    async with loopback_server() as server:
        api_url = f"{server.url}/api/v4"
        path = _path_for(api_url)
        async with path:
            first = await path.get("/api/v4/ping/one")
            second = await path.get("/api/v4/ping/two")
            connections = server.connection_count
            pooled_client = path.pooled_client if isinstance(path, _SharedPath) else None

    assert first.status_code == 200
    assert second.status_code == 200
    assert connections == 1
    if pooled_client is not None:
        assert pooled_client.is_closed


@pytest.mark.asyncio
async def test_client_per_call_control_cannot_reuse_a_connection() -> None:
    """The fixture detects non-reuse: a fresh client per request opens two."""
    async with loopback_server() as server:
        api_url = f"{server.url}/api/v4"
        async with _LegacyClientPerCall(api_url) as path:
            await path.get("/api/v4/legacy/one")
            await path.get("/api/v4/legacy/two")
            connections = server.connection_count

    assert connections >= 2


@pytest.mark.asyncio
async def test_concurrent_jobs_open_at_most_eight_connections() -> None:
    async with loopback_server(gate_target=8) as server:
        api_url = f"{server.url}/api/v4"
        async with _SharedPath(api_url) as path:
            responses = await asyncio.gather(
                *(path.get(f"/api/v4/concurrent/{index}") for index in range(16))
            )
            peak_connections = server.connection_count

    assert all(response.status_code == 200 for response in responses)
    assert peak_connections <= 8
    assert peak_connections >= 2
