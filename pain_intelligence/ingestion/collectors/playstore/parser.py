"""Parser for Google Play Store public review data (via google-play-scraper)."""

from __future__ import annotations

from typing import Any


class PlayStoreParser:
    """Extracts structured data from google-play-scraper response dicts."""

    @staticmethod
    def parse_reviews_page(
        reviews: list[dict[str, Any]],
        continuation_token: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Parse a page of reviews from google-play-scraper.

        Args:
            reviews: List of review dicts from the scraper.
            continuation_token: Token for next page, or None if exhausted.

        Returns:
            (review_items, continuation_token) tuple.
        """
        return reviews, continuation_token if continuation_token else None

    @staticmethod
    def parse_app_info(app_data: dict[str, Any]) -> dict[str, Any]:
        """Parse app metadata from google-play-scraper app() call.

        Returns a flat dict with standardized field names.
        """
        if not app_data:
            return {}

        # Extract categories from the genre/genreId or categories field
        categories = []
        if "genre" in app_data and app_data["genre"]:
            categories.append(app_data["genre"])
        if "categories" in app_data and isinstance(app_data["categories"], list):
            for cat in app_data["categories"]:
                if isinstance(cat, dict) and "name" in cat:
                    categories.append(cat["name"])
                elif isinstance(cat, str):
                    categories.append(cat)

        return {
            "title": app_data.get("title"),
            "package_name": app_data.get("appId"),
            "category": app_data.get("genre"),
            "categories": categories,
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
            "updated": app_data.get("lastUpdatedOn"),
            "icon": app_data.get("icon"),
            "header_image": app_data.get("headerImage"),
            "histogram": app_data.get("histogram"),
        }

    @staticmethod
    def extract_reviewer_name(review: dict[str, Any]) -> str:
        """Extract reviewer display name from a review."""
        return review.get("userName", "")

    @staticmethod
    def extract_review_text(review: dict[str, Any]) -> str:
        """Extract review text content."""
        return review.get("content", "")
