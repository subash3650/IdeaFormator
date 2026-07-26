"""Adapter for transforming Google Play Store public review data into normalized dicts."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.utils import compute_checksum, compute_document_id, parse_timestamp


class PlayStoreAdapter(BaseAdapter):
    """Transforms google-play-scraper review and app data into normalized dicts."""

    @property
    def source(self) -> SourceType:
        return SourceType.PLAYSTORE

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single google-play-scraper item.

        Supports both review dicts and app info dicts.
        """
        if "score" in raw_response and "userName" in raw_response:
            return self._transform_review(raw_response)
        if "appId" in raw_response or "package_name" in raw_response:
            return self._transform_app_info(raw_response)
        return self._transform_review(raw_response)

    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of google-play-scraper items."""
        return [
            self.transform(item)
            for item in raw_responses
            if item and ("userName" in item or "appId" in item or "reviewId" in item)
        ]

    def _transform_review(self, review: dict[str, Any]) -> dict[str, Any]:
        """Transform a google-play-scraper review into a normalized dict."""
        external_id = review.get("reviewId", "")
        author = review.get("userName", "")
        star_rating = review.get("score", 0)
        content = review.get("content", "")

        tags = [f"rating:{star_rating}", "type:review"]

        metadata = {
            "play_review_id": external_id,
            "star_rating": star_rating,
            "thumbs_up_count": review.get("thumbsUpCount", 0),
            "app_version": review.get("appVersion") or review.get("reviewCreatedVersion"),
            "review_created_version": review.get("reviewCreatedVersion"),
            "developer_reply": review.get("replyContent"),
            "developer_reply_date": parse_timestamp(review.get("repliedAt")),
            "user_image": review.get("userImage"),
            "package_name": review.get("appId"),
            "country": review.get("country"),
            "language": review.get("lang"),
        }

        document_id = compute_document_id("playstore", external_id)
        checksum = compute_checksum(f"{author}\n{content}")

        return {
            "document_id": document_id,
            "source": SourceType.PLAYSTORE,
            "source_type": "review",
            "external_id": external_id,
            "title": None,
            "content": content,
            "author": author,
            "created_at": parse_timestamp(review.get("at")),
            "updated_at": parse_timestamp(review.get("repliedAt")),
            "language": None,
            "url": "",
            "tags": tags,
            "categories": [f"rating:{star_rating}"],
            "metadata": metadata,
            "raw_json": review,
            "checksum": checksum,
            "collector_version": self.version,
        }

    def _transform_app_info(self, app_data: dict[str, Any]) -> dict[str, Any]:
        """Transform google-play-scraper app info into a normalized dict."""
        package_name = app_data.get("appId") or app_data.get("package_name", "")
        title = app_data.get("title", "")
        document_id = compute_document_id("playstore", f"app:{package_name}")
        checksum = compute_checksum(f"{title}\n{package_name}")

        tags = ["type:app_info"]
        category = app_data.get("category")
        if category:
            tags.append(f"category:{category}")

        metadata = {
            "package_name": package_name,
            "title": title,
            "category": category,
            "categories": app_data.get("categories", []),
            "developer": app_data.get("developer"),
            "developer_id": app_data.get("developerId"),
            "developer_email": app_data.get("developerEmail"),
            "developer_website": app_data.get("developerWebsite"),
            "description": app_data.get("description"),
            "summary": app_data.get("summary"),
            "score": app_data.get("score"),
            "ratings": app_data.get("ratings"),
            "reviews": app_data.get("reviews"),
            "installs": app_data.get("installs"),
            "min_installs": app_data.get("minInstalls"),
            "real_installs": app_data.get("realInstalls"),
            "price": app_data.get("price"),
            "free": app_data.get("free"),
            "currency": app_data.get("currency"),
            "version": app_data.get("version"),
            "content_rating": app_data.get("contentRating"),
            "released": app_data.get("released"),
            "updated": app_data.get("updated") or app_data.get("lastUpdatedOn"),
            "icon": app_data.get("icon"),
            "header_image": app_data.get("headerImage"),
            "histogram": app_data.get("histogram"),
        }

        return {
            "document_id": document_id,
            "source": SourceType.PLAYSTORE,
            "source_type": "app_info",
            "external_id": package_name,
            "title": title,
            "content": app_data.get("description") or app_data.get("summary"),
            "author": app_data.get("developer"),
            "created_at": parse_timestamp(app_data.get("released")),
            "updated_at": parse_timestamp(app_data.get("updated") or app_data.get("lastUpdatedOn")),
            "language": None,
            "url": f"https://play.google.com/store/apps/details?id={package_name}",
            "tags": tags,
            "categories": [category] if category else [],
            "metadata": metadata,
            "raw_json": app_data,
            "checksum": checksum,
            "collector_version": self.version,
        }
