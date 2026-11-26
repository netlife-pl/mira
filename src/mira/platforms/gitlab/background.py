"""Tracked background jobs that borrow the shared GitLab HTTP runtime.

Dashboard indexing routes spawn ``asyncio`` tasks that outlive their request.
The server lifespan uses this registry so it never closes pooled clients under
an in-flight GitLab job. Only dashboard-spawned GitLab-capable work belongs
here; no generic task framework is intended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_tasks: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create a tracked task before it can escape the spawning request."""
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def drain() -> None:
    """Cancel tracked jobs and await their terminal cleanup.

    Mirrors the lifespan's cancel-and-await shutdown policy so a stuck fetch or
    LLM call cannot hold shutdown open; cancellation releases the shared
    runtime's permits without touching its cooldowns.
    """
    while _tasks:
        tasks = list(_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
