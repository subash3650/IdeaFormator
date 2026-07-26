"""Adapter for transforming Product Hunt API responses into normalized dicts."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.collectors.producthunt.parser import ProductHuntParser
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.utils import compute_checksum, compute_document_id, parse_timestamp


class ProductHuntAdapter(BaseAdapter):
    """Transforms Product Hunt post and comment nodes into normalized dicts."""

    def __init__(self) -> None:
        self._parser = ProductHuntParser()

    @property
    def source(self) -> SourceType:
        return SourceType.PRODUCTHUNT

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single Product Hunt post or comment node."""
        if "commentBody" in raw_response or ("body" in raw_response and "post" in raw_response and "createdAt" in raw_response):
            return self._transform_comment(raw_response)
        return self._transform_post(raw_response)

    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of Product Hunt nodes."""
        return [self.transform(item) for item in raw_responses if item and "id" in item]

    def _transform_post(self, post: dict[str, Any]) -> dict[str, Any]:
        """Transform a Product Hunt post node into a normalized dict."""
        external_id = str(post.get("id", ""))
        name = post.get("name", "")
        tagline = post.get("tagline", "") or ""
        description = post.get("description", "") or ""
        votes_count = post.get("votesCount", 0)
        comments_count = post.get("commentsCount", 0)
        website = post.get("website", "") or ""
        ph_url = post.get("url", "") or ""

        topics = self._parser.extract_topics(post)
        makers = self._parser.extract_makers(post)

        tags = [f"votes:{votes_count}", "type:post"]
        for topic in topics:
            tags.append(f"topic:{topic}")

        metadata = {
            "ph_id": post.get("id"),
            "tagline": tagline,
            "votes_count": votes_count,
            "comments_count": comments_count,
            "website": website,
            "ph_url": ph_url,
            "topics": topics,
            "makers": makers,
            "reviews_count": post.get("reviewsRating", 0),
            "thumbnail_url": (post.get("thumbnail", {}) or {}).get("url", ""),
        }

        content = description if description else tagline
        document_id = compute_document_id("producthunt", external_id)
        checksum = compute_checksum(f"{name}\n{content}")

        return {
            "document_id": document_id,
            "source": SourceType.PRODUCTHUNT,
            "source_type": "post",
            "external_id": external_id,
            "title": name,
            "content": content,
            "author": makers[0]["username"] if makers else "",
            "created_at": parse_timestamp(post.get("createdAt")),
            "updated_at": parse_timestamp(post.get("createdAt")),
            "language": None,
            "url": ph_url or website,
            "tags": tags,
            "categories": [f"topic:{t}" for t in topics],
            "metadata": metadata,
            "raw_json": post,
            "checksum": checksum,
            "collector_version": self.version,
        }

    def _transform_comment(self, comment: dict[str, Any]) -> dict[str, Any]:
        """Transform a Product Hunt comment node into a normalized dict."""
        external_id = str(comment.get("id", ""))
        body = comment.get("commentBody", "") or comment.get("body", "") or ""

        tags = ["type:comment"]

        post_node = comment.get("post", {})
        post_id = post_node.get("id", "") if isinstance(post_node, dict) else ""

        author_node = comment.get("author", {}) or {}
        author_name = author_node.get("name", "") or author_node.get("username", "")

        metadata = {
            "ph_id": comment.get("id"),
            "post_id": post_id,
            "author_name": author_node.get("name", ""),
            "author_username": author_node.get("username", ""),
        }

        document_id = compute_document_id("producthunt_comment", external_id)
        checksum = compute_checksum(body)

        return {
            "document_id": document_id,
            "source": SourceType.PRODUCTHUNT,
            "source_type": "comment",
            "external_id": external_id,
            "title": None,
            "content": body,
            "author": author_name,
            "created_at": parse_timestamp(comment.get("createdAt")),
            "updated_at": parse_timestamp(comment.get("createdAt")),
            "language": None,
            "url": "",
            "tags": tags,
            "categories": [],
            "metadata": metadata,
            "raw_json": comment,
            "checksum": checksum,
            "collector_version": self.version,
        }
