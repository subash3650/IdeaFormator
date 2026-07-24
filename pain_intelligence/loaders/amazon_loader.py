"""Amazon Reviews dataset loader.

Handles the Amazon Reviews CSV schema:
Reviewer Name, Profile Link, Country, Review Count, Review Date,
Rating, Review Title, Review Text, Date of Experience
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.loaders.registry import register_loader
from pain_intelligence.schema.document import Document, Platform

REQUIRED_COLUMNS = frozenset({"Reviewer Name", "Rating", "Review Text"})
ALL_COLUMNS = frozenset({
    "Reviewer Name", "Profile Link", "Country", "Review Count",
    "Review Date", "Rating", "Review Title", "Review Text",
    "Date of Experience",
})


@register_loader
class AmazonLoader(BaseLoader):
    """Loader for Amazon Reviews datasets."""

    platform = Platform.AMAZON
    source_name = "amazon_reviews"

    def _detect(self, df_columns: list[str]) -> bool:
        """Detect Amazon by presence of Reviewer Name + Review Text + Rating."""
        col_set = set(df_columns)
        return REQUIRED_COLUMNS.issubset(col_set)

    def _transform_row(self, row: dict[str, Any]) -> Document:
        """Transform an Amazon review row into a Document."""
        text = self._safe_str(row.get("Review Text"))
        if not text:
            raise ValueError("Empty review text")

        rating_str = self._safe_str(row.get("Rating")) or ""
        rating = self._extract_rating_number(rating_str)

        created_at = self._parse_date(self._safe_str(row.get("Review Date")))
        experience_date = self._safe_str(row.get("Date of Experience"))

        metadata: dict[str, Any] = {}
        if experience_date:
            metadata["date_of_experience"] = experience_date
        review_count = self._safe_str(row.get("Review Count"))
        if review_count:
            metadata["review_count"] = review_count

        return Document(
            platform=self.platform,
            source_dataset="Amazon_Reviews.csv",
            title=self._safe_str(row.get("Review Title")),
            text=text,
            rating=rating,
            author=self._safe_str(row.get("Reviewer Name")),
            country=self._safe_str(row.get("Country")),
            created_at=created_at,
            metadata=metadata,
            raw_record={k: str(v) for k, v in row.items() if v is not None},
        )

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        """Parse ISO format date string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
