"""Twitter dataset loader.

Handles two Twitter CSV schemas:
1. business_id, Location, type, text  (Twitter_Data.csv)
2. clean_text, category               (twitter_training.csv)
"""

from __future__ import annotations

from typing import Any

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.loaders.registry import register_loader
from pain_intelligence.schema.document import Document, Platform

SCHEMA_1_COLUMNS = frozenset({"business_id", "Location", "type", "text"})
SCHEMA_2_COLUMNS = frozenset({"clean_text", "category"})


@register_loader
class TwitterLoader(BaseLoader):
    """Loader for Twitter datasets (both schema variants)."""

    platform = Platform.TWITTER
    source_name = "twitter"

    def _detect(self, df_columns: list[str]) -> bool:
        """Detect Twitter by column patterns."""
        col_set = set(df_columns)
        if SCHEMA_1_COLUMNS.issubset(col_set):
            return True
        if SCHEMA_2_COLUMNS.issubset(col_set):
            return True
        return False

    def _transform_row(self, row: dict[str, Any]) -> Document:
        """Transform a Twitter row into a Document."""
        columns = set(row.keys())

        if "text" in columns and "Location" in columns:
            return self._transform_schema1(row)
        elif "clean_text" in columns and "category" in columns:
            return self._transform_schema2(row)
        else:
            raise ValueError(f"Unknown Twitter schema: {list(columns)}")

    def _transform_schema1(self, row: dict[str, Any]) -> Document:
        """Transform Twitter_Data.csv format."""
        text = self._safe_str(row.get("text"))
        if not text:
            raise ValueError("Empty text")

        location = self._safe_str(row.get("Location"))
        sentiment = self._safe_str(row.get("type"))

        metadata: dict[str, Any] = {
            "business_id": str(row.get("business_id", "")),
        }
        if sentiment:
            metadata["sentiment_label"] = sentiment

        return Document(
            platform=self.platform,
            source_dataset="Twitter_Data.csv",
            text=text,
            location=location,
            metadata=metadata,
            raw_record={k: str(v) for k, v in row.items() if v is not None},
        )

    def _transform_schema2(self, row: dict[str, Any]) -> Document:
        """Transform twitter_training.csv format."""
        text = self._safe_str(row.get("clean_text"))
        if not text:
            raise ValueError("Empty text")

        category = self._safe_str(row.get("category"))

        metadata: dict[str, Any] = {}
        if category is not None:
            metadata["sentiment_category"] = category

        return Document(
            platform=self.platform,
            source_dataset="twitter_training.csv",
            text=text,
            metadata=metadata,
            raw_record={k: str(v) for k, v in row.items() if v is not None},
        )
