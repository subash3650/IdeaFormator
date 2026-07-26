"""httpx-based HTTP client implementation."""

from __future__ import annotations

from typing import Any

import httpx

from pain_intelligence.ingestion.clients.base import HttpClient


class HttpxClient(HttpClient):
    """Concrete HTTP client using httpx in synchronous mode."""

    def __init__(self, timeout: int = 30) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"Accept": "application/json", "User-Agent": "IdeaFormator-Ingestion/0.1.0"},
        )

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._client.get(url, params=params, headers=headers)

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._client.post(url, json=json, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpxClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
