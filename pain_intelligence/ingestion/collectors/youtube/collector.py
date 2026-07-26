"""Collector for YouTube Data API v3 with quota-aware behavior."""

from __future__ import annotations

import os
from typing import Any, Iterator

from pain_intelligence.ingestion.collectors.base import BaseCollector, COLLECTOR_VERSION
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.collectors.youtube.adapter import YouTubeAdapter
from pain_intelligence.ingestion.collectors.youtube.parser import YouTubeParser
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.registry import register_collector
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

YT_API_BASE = "https://www.googleapis.com/youtube/v3"

# Default search queries for tech/startup content
DEFAULT_QUERIES = [
    "artificial intelligence startup",
    "developer tools launch",
    "open source project",
    "tech product launch",
]

# Quota costs per endpoint (YouTube Data API v3)
QUOTA_COST_SEARCH = 100  # search.list
QUOTA_COST_VIDEOS = 1   # videos.list
QUOTA_COST_COMMENTS = 1  # commentThreads.list

# Daily quota limit (conservative estimate)
DEFAULT_DAILY_QUOTA = 10000


@register_collector("youtube")
class YouTubeCollector(BaseCollector):
    """Collects videos, comments, and channel data from YouTube Data API v3.

    Supports:
    - Quota-aware fetching with graceful exhaustion handling
    - Search-based video discovery with configurable queries
    - Comment collection per video
    - Incremental sync via publishedAfter parameter
    """

    adapter_class = YouTubeAdapter

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        super().__init__(config, client)
        self._api_key: str | None = None
        self._parser = YouTubeParser()
        self._quota_remaining: int = DEFAULT_DAILY_QUOTA

    @property
    def source(self) -> SourceType:
        return SourceType.YOUTUBE

    def authenticate(self) -> None:
        """Set up YouTube API key."""
        if self._config.api_key_env:
            self._api_key = os.environ.get(self._config.api_key_env)

    def health_check(self) -> bool:
        """Verify YouTube API is reachable via a channels.list probe."""
        try:
            params = {
                "part": "snippet",
                "forUsername": "YouTube",
                "key": self._api_key or "",
            }
            response = self._client.get(f"{YT_API_BASE}/channels", params=params)
            self._api_calls += 1
            self._quota_remaining -= QUOTA_COST_VIDEOS

            data = response.json()
            if "error" in data:
                error_code = data["error"].get("code", 0)
                if error_code in (400, 403):
                    logger.warning("[youtube] API key invalid or quota exhausted.")
                    return False

            if response.status_code == 200 and "items" in data:
                logger.info("[youtube] API healthy. Quota remaining: ~{}", self._quota_remaining)
                return True

            logger.warning("[youtube] Health check returned status {}", response.status_code)
            return False
        except Exception as e:
            logger.error("[youtube] Health check failed: {}", e)
            return False

    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch videos from YouTube, yielding batches per search query.

        Each batch is a list of video items or comment thread items.
        """
        if self._quota_remaining <= 0:
            logger.warning("[youtube] Quota exhausted. Skipping fetch.")
            return

        yield from self._fetch_videos(state)

    def _fetch_videos(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch videos via search.list + videos.list with quota tracking."""
        queries = DEFAULT_QUERIES
        batch_size = min(self._config.batch_size, 50)  # YouTube max is 50
        max_pages = self._config.max_pages
        published_after = state.last_sync.isoformat() if state and state.last_sync else None

        for query in queries:
            if self._quota_remaining <= QUOTA_COST_SEARCH + QUOTA_COST_VIDEOS:
                logger.warning("[youtube] Quota nearly exhausted. Stopping.")
                return

            yield from self._search_videos(query, batch_size, max_pages, published_after)

    def _search_videos(
        self,
        query: str,
        batch_size: int,
        max_pages: int,
        published_after: str | None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Search for videos and fetch their details."""
        next_token: str | None = None

        for page in range(max_pages):
            if self._quota_remaining < QUOTA_COST_SEARCH:
                return

            # Search for video IDs
            search_params: dict[str, Any] = {
                "part": "id",
                "q": query,
                "type": "video",
                "order": "date",
                "maxResults": batch_size,
                "key": self._api_key or "",
            }
            if published_after:
                search_params["publishedAfter"] = published_after
            if next_token:
                search_params["pageToken"] = next_token

            logger.debug("[youtube] Search query '{}' page {}", query, page + 1)
            response = self._client.get(f"{YT_API_BASE}/search", params=search_params)
            self._api_calls += 1
            self._quota_remaining -= QUOTA_COST_SEARCH

            if response.status_code != 200:
                logger.warning("[youtube] Search returned {}: {}", response.status_code, response.text[:200])
                break

            data = response.json()
            if "error" in data:
                error_code = data["error"].get("code", 0)
                if error_code == 403:
                    logger.warning("[youtube] Quota exhausted during search.")
                    return
                logger.warning("[youtube] API error: {}", data["error"].get("message", ""))
                break

            search_items, next_token = self._parser.parse_search_results(data)

            if not search_items:
                break

            # Extract video IDs
            video_ids = [
                item.get("id", {}).get("videoId", "")
                for item in search_items
                if item.get("id", {}).get("kind") == "youtube#video"
            ]
            video_ids = [vid for vid in video_ids if vid]

            if not video_ids:
                if not next_token:
                    break
                continue

            # Fetch video details
            if self._quota_remaining < QUOTA_COST_VIDEOS:
                return

            video_params: dict[str, Any] = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids[:batch_size]),
                "key": self._api_key or "",
            }
            video_response = self._client.get(f"{YT_API_BASE}/videos", params=video_params)
            self._api_calls += 1
            self._quota_remaining -= QUOTA_COST_VIDEOS

            if video_response.status_code != 200:
                logger.warning("[youtube] Videos list returned {}", video_response.status_code)
                break

            video_data = video_response.json()
            videos = self._parser.parse_videos_response(video_data)

            if videos:
                yield videos

                # Fetch comments for each video (limited)
                for video in videos[:3]:  # Limit comment fetching
                    video_id = video.get("id", "")
                    if video_id and self._quota_remaining >= QUOTA_COST_COMMENTS:
                        comments = self._fetch_video_comments(video_id)
                        if comments:
                            yield comments

            if not next_token:
                break

    def _fetch_video_comments(self, video_id: str) -> list[dict[str, Any]]:
        """Fetch comments for a single video."""
        all_comments: list[dict[str, Any]] = []
        next_token: str | None = None

        for _ in range(2):  # Limit comment pages per video
            if self._quota_remaining < QUOTA_COST_COMMENTS:
                break

            params: dict[str, Any] = {
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",
                "maxResults": 20,
                "key": self._api_key or "",
            }
            if next_token:
                params["pageToken"] = next_token

            response = self._client.get(f"{YT_API_BASE}/commentThreads", params=params)
            self._api_calls += 1
            self._quota_remaining -= QUOTA_COST_COMMENTS

            if response.status_code != 200:
                # Comments might be disabled
                break

            data = response.json()
            if "error" in data:
                break

            comments, next_token = self._parser.parse_comment_threads(data)
            all_comments.extend(comments)

            if not next_token:
                break

        return all_comments
