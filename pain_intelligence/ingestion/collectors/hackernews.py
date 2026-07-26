"""Collector for Hacker News via the official Firebase API."""

from __future__ import annotations

from typing import Any, Iterator

from pain_intelligence.ingestion.adapters.hackernews import HackerNewsAdapter
from pain_intelligence.ingestion.collectors.base import BaseCollector, COLLECTOR_VERSION
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.registry import register_collector
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# HN story endpoints
STORY_ENDPOINTS = ["topstories", "newstories", "beststories"]


@register_collector("hackernews")
class HackerNewsCollector(BaseCollector):
    """Collects stories and comments from Hacker News via the Firebase API.

    Supports:
    - Multiple story lists (top, new, best)
    - Pagination via start/end index slicing
    - Individual item fetching for comments
    - Incremental sync via item ID tracking
    """

    adapter_class = HackerNewsAdapter

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        super().__init__(config, client)
        self._base_url = HN_API_BASE

    @property
    def source(self) -> SourceType:
        return SourceType.HACKERNEWS

    def authenticate(self) -> None:
        """No authentication needed for HN Firebase API."""
        pass

    def health_check(self) -> bool:
        """Verify HN Firebase API is reachable."""
        try:
            response = self._client.get(f"{self._base_url}/maxitem.json")
            self._api_calls += 1
            if response.status_code == 200:
                max_id = response.json()
                logger.info("[hackernews] API healthy. Max item ID: {}", max_id)
                return True
            logger.warning("[hackernews] Health check returned status {}", response.status_code)
            return False
        except Exception as e:
            logger.error("[hackernews] Health check failed: {}", e)
            return False

    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch stories from HN, yielding batches per story endpoint.

        Each batch is a list of fully hydrated item dicts (stories + their comments).
        """
        batch_size = self._config.batch_size
        max_pages = self._config.max_pages
        items_per_page = batch_size * 5  # Fetch more IDs than needed, some may fail

        for endpoint in STORY_ENDPOINTS:
            yield from self._fetch_story_list(endpoint, items_per_page, max_pages, state)

    def _fetch_story_list(
        self,
        endpoint: str,
        items_per_page: int,
        max_pages: int,
        state: SyncState | None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Fetch stories from a single HN endpoint (top/new/best)."""
        # Get the list of story IDs
        url = f"{self._base_url}/{endpoint}.json"
        response = self._client.get(url)
        self._api_calls += 1

        if response.status_code != 200:
            logger.warning("[hackernews] Failed to fetch {}: {}", endpoint, response.status_code)
            return

        story_ids = response.json()
        if not story_ids:
            return

        logger.info("[hackernews] Found {} stories in {}", len(story_ids), endpoint)

        # Paginate through story IDs
        start = 0
        for page in range(max_pages):
            end = min(start + items_per_page, len(story_ids))
            batch_ids = story_ids[start:end]

            if not batch_ids:
                break

            # Fetch full item details for each story
            batch_items: list[dict[str, Any]] = []
            for item_id in batch_ids:
                item = self._fetch_item(item_id)
                if item and not item.get("deleted") and not item.get("dead"):
                    batch_items.append(item)

                    # Optionally fetch top-level comments for the story
                    kids = item.get("kids", [])
                    if kids:
                        comments = self._fetch_comments(kids[:5])  # Limit to first 5 comments
                        batch_items.extend(comments)

            if batch_items:
                yield batch_items

            start = end
            if start >= len(story_ids):
                break

    def _fetch_item(self, item_id: int) -> dict[str, Any] | None:
        """Fetch a single HN item by ID."""
        url = f"{self._base_url}/item/{item_id}.json"
        try:
            response = self._client.get(url)
            self._api_calls += 1
            if response.status_code == 200:
                return response.json()
            logger.debug("[hackernews] Item {} returned {}", item_id, response.status_code)
            return None
        except Exception as e:
            logger.debug("[hackernews] Failed to fetch item {}: {}", item_id, e)
            return None

    def _fetch_comments(self, item_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch multiple comment items."""
        comments: list[dict[str, Any]] = []
        for item_id in item_ids:
            item = self._fetch_item(item_id)
            if item and item.get("type") == "comment" and not item.get("deleted") and not item.get("dead"):
                text = item.get("text", "")
                if text:  # Skip empty/deleted comments
                    comments.append(item)
        return comments
