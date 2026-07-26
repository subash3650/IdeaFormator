"""Parser for YouTube Data API v3 responses."""

from __future__ import annotations

from typing import Any


class YouTubeParser:
    """Extracts structured data from YouTube Data API v3 response payloads."""

    @staticmethod
    def parse_search_results(response_data: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Parse a search.list response.

        Returns (video_items, nextPageToken) tuple.
        """
        items = response_data.get("items", [])
        next_token = response_data.get("nextPageToken")
        return items, next_token

    @staticmethod
    def parse_videos_response(response_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse a videos.list response."""
        return response_data.get("items", [])

    @staticmethod
    def parse_comment_threads(response_data: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Parse a commentThreads.list response.

        Returns (comment_items, nextPageToken) tuple.
        """
        items = response_data.get("items", [])
        next_token = response_data.get("nextPageToken")
        return items, next_token

    @staticmethod
    def parse_channel_response(response_data: dict[str, Any]) -> dict[str, Any] | None:
        """Parse a channels.list response. Returns first channel item or None."""
        items = response_data.get("items", [])
        return items[0] if items else None

    @staticmethod
    def parse_quota_info(response_data: dict[str, Any]) -> int | None:
        """Extract quota cost hint from response headers (if available)."""
        return None  # Quota info comes from response headers, not body
