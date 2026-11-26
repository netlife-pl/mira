"""Per-runtime tracking of background jobs that borrow a GitLab HTTP runtime.

Dashboard indexing routes spawn ``asyncio`` tasks that outlive their request.
Each owning ``GitLabHttpRuntime`` gets its own task set, so shutting down one
application can never cancel or await another application's jobs, and a
drain only ever gathers tasks owned by that runtime's loop. No generic task
framework is intended.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Coroutine
from typing import Any

from mira.platforms.gitlab.http import _RUNTIME_CONTEXT

_owner_tasks: weakref.WeakKeyDictionary[Any, set[asyncio.Task[Any]]] = weakref.WeakKeyDictionary()


def _bucket(owner: Any) -> set[asyncio.Task[Any]]:
    tasks = _owner_tasks.get(owner)
    if tasks is None:
        tasks = set()
        _owner_tasks[owner] = tasks
    return tasks


def _forget(owner: Any, task: asyncio.Task[Any]) -> None:
    tasks = _owner_tasks.get(owner)
    if tasks is not None:
        tasks.discard(task)


def track(
    task: asyncio.Task[Any],
    *,
    owner: Any | None = None,
    request: Any | None = None,
) -> asyncio.Task[Any]:
    """Associate ``task`` with its owning runtime before it can escape.

    ``owner`` is explicit when the caller has it; otherwise the request's
    lifespan runtime is used, then the bound runtime scope. Tasks with no
    discoverable owner are left untracked rather than assigned globally.
    """
    if owner is None and request is not None:
        owner = getattr(request.app.state, "gitlab_http_runtime", None)
    if owner is None:
        owner = _RUNTIME_CONTEXT.get()
    if owner is None:
        return task
    _bucket(owner).add(task)
    task.add_done_callback(lambda done: _forget(owner, done))
    return task


def spawn(
    coro: Coroutine[Any, Any, Any],
    *,
    owner: Any | None = None,
    request: Any | None = None,
) -> asyncio.Task[Any]:
    """Create a task owned by the calling request/lifespan runtime."""
    return track(asyncio.create_task(coro), owner=owner, request=request)


def track_current(*, owner: Any | None = None) -> None:
    """Track the currently running task (self-registering batch functions)."""
    task = asyncio.current_task()
    if task is not None:
        track(task, owner=owner)


async def drain(owner: Any) -> None:
    """Cancel and await only ``owner``'s tracked jobs.

    Mirrors the lifespan's cancel-and-await shutdown policy so a stuck fetch or
    LLM call cannot hold shutdown open; cancellation releases the shared
    runtime's permits without touching its cooldowns.
    """
    while True:
        tasks = list(_owner_tasks.get(owner, ()))
        if not tasks:
            return
        bucket = _bucket(owner)
        for task in tasks:
            bucket.discard(task)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
