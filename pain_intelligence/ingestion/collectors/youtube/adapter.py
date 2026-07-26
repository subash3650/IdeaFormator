"""Adapter for transforming YouTube Data API v3 responses into normalized dicts."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.utils import compute_checksum, compute_document_id, parse_timestamp


class YouTubeAdapter(BaseAdapter):
    """Transforms YouTube video and comment API responses into normalized dicts."""

    @property
    def source(self) -> SourceType:
        return SourceType.YOUTUBE

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single YouTube video or comment item."""
        kind = raw_response.get("kind", "")
        if "comment" in kind or ("snippet" in raw_response and "topLevelComment" in raw_response.get("snippet", {})):
            return self._transform_comment_thread(raw_response)
        return self._transform_video(raw_response)

    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of YouTube items."""
        return [self.transform(item) for item in raw_responses if item and "id" in item]

    def _transform_video(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform a YouTube video item into a normalized dict."""
        video_id = item.get("id", "")
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})

        title = snippet.get("title", "")
        description = snippet.get("description", "") or ""
        channel_title = snippet.get("channelTitle", "")
        channel_id = snippet.get("channelId", "")
        published_at = snippet.get("publishedAt")
        tags = snippet.get("tags", []) or []
        category_id = snippet.get("categoryId", "")

        view_count = int(statistics.get("viewCount", 0))
        like_count = int(statistics.get("likeCount", 0))
        comment_count = int(statistics.get("commentCount", 0))

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        meta_tags = [f"views:{view_count}", f"likes:{like_count}", "type:video"]
        if category_id:
            meta_tags.append(f"category:{category_id}")

        metadata = {
            "yt_id": video_id,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "duration": content_details.get("duration", ""),
            "definition": content_details.get("definition", ""),
            "caption": content_details.get("caption", "false"),
            "tags": tags,
            "category_id": category_id,
            "yt_url": video_url,
        }

        content = description if description else title
        document_id = compute_document_id("youtube", video_id)
        checksum = compute_checksum(f"{title}\n{content}")

        return {
            "document_id": document_id,
            "source": SourceType.YOUTUBE,
            "source_type": "video",
            "external_id": video_id,
            "title": title,
            "content": content,
            "author": channel_title,
            "created_at": parse_timestamp(published_at),
            "updated_at": parse_timestamp(snippet.get("publishedAt")),
            "language": snippet.get("defaultLanguage"),
            "url": video_url,
            "tags": meta_tags,
            "categories": [f"channel:{channel_id}"] if channel_id else [],
            "metadata": metadata,
            "raw_json": item,
            "checksum": checksum,
            "collector_version": self.version,
        }

    def _transform_comment_thread(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform a YouTube comment thread into a normalized dict."""
        snippet = item.get("snippet", {})
        top_comment = snippet.get("topLevelComment", {})
        comment_snippet = top_comment.get("snippet", {})

        comment_id = top_comment.get("id", "") or item.get("id", "")
        text = comment_snippet.get("textDisplay", "") or comment_snippet.get("textOriginal", "") or ""
        author = comment_snippet.get("authorDisplayName", "")
        video_id = snippet.get("videoId", "")

        tags = ["type:comment"]

        metadata = {
            "yt_comment_id": comment_id,
            "video_id": video_id,
            "author_channel_id": comment_snippet.get("authorChannelId", {}).get("value", ""),
            "like_count": int(comment_snippet.get("likeCount", 0)),
            "total_reply_count": snippet.get("totalReplyCount", 0),
            "author_display_name": author,
        }

        document_id = compute_document_id("youtube_comment", comment_id)
        checksum = compute_checksum(text)

        return {
            "document_id": document_id,
            "source": SourceType.YOUTUBE,
            "source_type": "comment",
            "external_id": comment_id,
            "title": None,
            "content": text,
            "author": author,
            "created_at": parse_timestamp(comment_snippet.get("publishedAt")),
            "updated_at": parse_timestamp(comment_snippet.get("updatedAt")),
            "language": comment_snippet.get("textOriginal", "")[:0] or None,
            "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
            "tags": tags,
            "categories": [],
            "metadata": metadata,
            "raw_json": item,
            "checksum": checksum,
            "collector_version": self.version,
        }
