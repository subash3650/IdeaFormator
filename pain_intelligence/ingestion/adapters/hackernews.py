"""Adapter for transforming Hacker News Firebase API responses into normalized dicts."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.utils import compute_checksum, compute_document_id, parse_timestamp


class HackerNewsAdapter(BaseAdapter):
    """Transforms HN Firebase API item dicts into normalized dicts."""

    @property
    def source(self) -> SourceType:
        return SourceType.HACKERNEWS

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single HN item dict into a normalized dict."""
        item_type = raw_response.get("type", "story")

        if item_type == "comment":
            return self._transform_comment(raw_response)
        return self._transform_story(raw_response)

    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of HN items. Filters out empty or id-less dicts."""
        return [self.transform(item) for item in raw_responses if item and "id" in item and "type" in item]

    def _transform_story(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform a HN story/ask/show/job into a normalized dict."""
        external_id = str(item.get("id", ""))
        title = item.get("title", "")
        text = item.get("text", "") or ""
        url = item.get("url", "")
        score = item.get("score", 0)
        descendants = item.get("descendants", 0)  # comment count
        item_type = item.get("type", "story")

        # Determine source_type from HN item type
        if item_type == "story":
            if title.lower().startswith("ask hn:"):
                source_type = "ask_hn"
            elif title.lower().startswith("show hn:"):
                source_type = "show_hn"
            else:
                source_type = "story"
        else:
            source_type = item_type

        # Tags from metadata
        tags = [f"score:{score}", f"type:{source_type}"]
        if url:
            tags.append("has_url")

        metadata = {
            "hn_id": item.get("id"),
            "score": score,
            "descendants": descendants,
            "item_type": item_type,
            "hn_url": f"https://news.ycombinator.com/item?id={external_id}",
            "parent_id": item.get("parent"),
            "kids": item.get("kids", []),
            "dead": item.get("dead", False),
            "deleted": item.get("deleted", False),
        }

        # Prefer URL content, fall back to text
        content = text if text else url
        document_id = compute_document_id("hackernews", external_id)
        checksum = compute_checksum(f"{title}\n{content}")

        return {
            "document_id": document_id,
            "source": SourceType.HACKERNEWS,
            "source_type": source_type,
            "external_id": external_id,
            "title": title,
            "content": content,
            "author": item.get("by", ""),
            "created_at": parse_timestamp(item.get("time")),
            "updated_at": None,  # HN API doesn't provide updated_at
            "language": "en",  # HN is predominantly English
            "url": metadata["hn_url"],
            "tags": tags,
            "categories": [],
            "metadata": metadata,
            "raw_json": item,
            "checksum": checksum,
            "collector_version": self.version,
        }

    def _transform_comment(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform a HN comment into a normalized dict."""
        external_id = str(item.get("id", ""))
        text = item.get("text", "") or ""

        tags = ["type:comment"]

        metadata = {
            "hn_id": item.get("id"),
            "item_type": "comment",
            "hn_url": f"https://news.ycombinator.com/item?id={external_id}",
            "parent_id": item.get("parent"),
            "kids": item.get("kids", []),
            "dead": item.get("dead", False),
            "deleted": item.get("deleted", False),
        }

        document_id = compute_document_id("hackernews_comment", external_id)
        checksum = compute_checksum(text)

        return {
            "document_id": document_id,
            "source": SourceType.HACKERNEWS,
            "source_type": "comment",
            "external_id": external_id,
            "title": None,
            "content": text,
            "author": item.get("by", ""),
            "created_at": parse_timestamp(item.get("time")),
            "updated_at": None,
            "language": "en",
            "url": metadata["hn_url"],
            "tags": tags,
            "categories": [],
            "metadata": metadata,
            "raw_json": item,
            "checksum": checksum,
            "collector_version": self.version,
        }
