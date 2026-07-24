"""Yelp Reviews dataset loader.

Handles the Yelp CSV schema:
business_id, date, review_id, stars, text, type, user_id, cool, useful, funny
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.loaders.registry import register_loader
from pain_intelligence.schema.document import Document, Platform

REQUIRED_COLUMNS = frozenset({"review_id", "stars", "text"})


@register_loader
class YelpLoader(BaseLoader):
    """Loader for Yelp Reviews datasets."""

    platform = Platform.YELP
    source_name = "yelp_reviews"

    def _detect(self, df_columns: list[str]) -> bool:
        """Detect Yelp by review_id + stars + text columns."""
        col_set = set(df_columns)
        return REQUIRED_COLUMNS.issubset(col_set)

    def _transform_row(self, row: dict[str, Any]) -> Document:
        """Transform a Yelp review row into a Document."""
        text = self._safe_str(row.get("text"))
        if not text:
            raise ValueError("Empty review text")

        rating = self._safe_float(row.get("stars"))

        created_at = self._parse_date(self._safe_str(row.get("date")))

        metadata: dict[str, Any] = {
            "business_id": str(row.get("business_id", "")),
            "review_id": str(row.get("review_id", "")),
            "user_id": str(row.get("user_id", "")),
            "type": str(row.get("type", "")),
            "cool": row.get("cool", 0),
            "useful": row.get("useful", 0),
            "funny": row.get("funny", 0),
        }

        return Document(
            platform=self.platform,
            source_dataset="yelp.csv",
            text=text,
            rating=rating,
            created_at=created_at,
            metadata=metadata,
            raw_record={k: str(v) for k, v in row.items() if v is not None},
        )

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        """Parse Yelp date format (YYYY-MM-DD)."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        return None
