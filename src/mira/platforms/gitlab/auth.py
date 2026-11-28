"""GitLab authentication — a static group/project access token."""

from __future__ import annotations

import httpx

from mira.http import (
    HttpRequestError,
    HttpRetryPolicy,
    HttpService,
    current_http_service,
    http_service_scope,
)


class GitLabTokenAuth:
    """A static group/project access token. No minting, no expiry handling."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://gitlab.com/api/v4",
        http_service: HttpService | None = None,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._http_service = http_service
        self._username_fetched = False
        self._username: str | None = None

    async def get_token(self, scope: str | int | None = None) -> str:
        return self._token

    async def _username_from_api(self) -> str | None:
        service = self._http_service or current_http_service()
        if service is None:
            async with http_service_scope() as scoped:
                return await self._username_with_service(scoped)
        return await self._username_with_service(service)

    async def _username_with_service(self, service: HttpService) -> str | None:
        response = await service.request(
            f"{self._base_url}/user",
            method="GET",
            credential=self._token,
            headers={"PRIVATE-TOKEN": self._token},
            timeout=10.0,
            policy=HttpRetryPolicy(max_attempts=3, deadline_seconds=60.0),
        )
        if response.status_code != 200:
            return None
        username = response.json().get("username")
        return username if isinstance(username, str) and username else None

    async def get_bot_identity(self) -> str | None:
        """The token user's username, via ``GET /user`` (cached)."""
        if self._username_fetched:
            return self._username
        self._username_fetched = True
        try:
            self._username = await self._username_from_api()
        except (httpx.HTTPError, ValueError, HttpRequestError):
            self._username = None
        return self._username
