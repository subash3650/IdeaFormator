"""Collector for GitHub Issues and Comments via REST API v3."""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from pain_intelligence.ingestion.adapters.github import GitHubAdapter
from pain_intelligence.ingestion.collectors.base import BaseCollector, COLLECTOR_VERSION
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.registry import register_collector
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

# Default repos to collect from — can be extended via config
DEFAULT_REPOS = [
    "torvalds/linux",
    "microsoft/vscode",
    "facebook/react",
    "golang/go",
    "rust-lang/rust",
]


@register_collector("github")
class GitHubCollector(BaseCollector):
    """Collects GitHub Issues via the REST API v3.

    Supports:
    - Pagination via Link header / page parameter
    - Incremental sync via `since` parameter (ISO 8601 timestamp)
    - Configurable repo list (falls back to DEFAULT_REPOS)
    """

    adapter_class = GitHubAdapter

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        super().__init__(config, client)
        self._token: str | None = None
        self._repos: list[str] = config.batch_size and DEFAULT_REPOS  # Override via config if needed
        self._base_url = "https://api.github.com"

    @property
    def source(self) -> SourceType:
        return SourceType.GITHUB

    def authenticate(self) -> None:
        """Set up GitHub API token."""
        self._token = self._config.api_key_env and __import__("os").environ.get(self._config.api_key_env)
        # Token is passed per-request via headers in _request()

    def health_check(self) -> bool:
        """Verify GitHub API is reachable."""
        try:
            headers = self._build_headers()
            response = self._client.get(f"{self._base_url}/rate_limit", headers=headers)
            self._api_calls += 1
            if response.status_code == 200:
                data = response.json()
                remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
                logger.info("[github] API healthy. Rate limit remaining: {}", remaining)
                return True
            logger.warning("[github] Health check returned status {}", response.status_code)
            return response.status_code == 200
        except Exception as e:
            logger.error("[github] Health check failed: {}", e)
            return False

    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch issues from configured repositories.

        Yields one batch per page per repo. Each batch is a list of issue dicts.
        """
        repos = self._get_repos()
        headers = self._build_headers()

        for repo in repos:
            yield from self._fetch_repo_issues(repo, headers, state)

    def _fetch_repo_issues(
        self,
        repo: str,
        headers: dict[str, str],
        state: SyncState | None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Fetch issues from a single repository with pagination."""
        page = 1
        max_pages = self._config.max_pages
        per_page = min(self._config.batch_size, 100)  # GitHub max is 100

        params: dict[str, Any] = {
            "state": "all",
            "per_page": per_page,
            "sort": "updated",
            "direction": "desc",
        }

        # Incremental sync: only fetch issues updated since last sync
        if state and state.last_sync:
            params["since"] = state.last_sync.isoformat()

        while page <= max_pages:
            params["page"] = page
            url = f"{self._base_url}/repos/{repo}/issues"

            logger.debug("[github] GET {}?page={}", url, page)
            response = self._client.get(url, params=params, headers=headers)
            self._api_calls += 1

            if response.status_code != 200:
                logger.warning(
                    "[github] GET {} returned {}: {}",
                    repo,
                    response.status_code,
                    response.text[:200],
                )
                break

            issues = response.json()
            if not issues:
                break

            yield issues

            # Check for next page via Link header
            if not self._has_next_page(response):
                break

            page += 1

        # Also fetch comments for issues in the latest batch
        # (optional: can be enabled via config)

    def _has_next_page(self, response: httpx.Response) -> bool:
        """Check if the response Link header contains a 'next' page."""
        link_header = response.headers.get("Link", "")
        return 'rel="next"' in link_header

    def _get_repos(self) -> list[str]:
        """Return list of repos to collect from."""
        # In a full implementation, this would read from config.
        # For now, use DEFAULT_REPOS.
        return DEFAULT_REPOS

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with optional auth token."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        return headers
