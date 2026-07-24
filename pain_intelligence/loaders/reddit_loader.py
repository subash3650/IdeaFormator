"""Reddit dataset loader.

Handles the Reddit CSV schema:
clean_comment, category
"""

from __future__ import annotations

from typing import Any

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.loaders.registry import register_loader
from pain_intelligence.schema.document import Document, Platform

REQUIRED_COLUMNS = frozenset({"clean_comment", "category"})


@register_loader
class RedditLoader(BaseLoader):
    """Loader for Reddit datasets."""

    platform = Platform.REDDIT
    source_name = "reddit"

    def _detect(self, df_columns: list[str]) -> bool:
        """Detect Reddit by clean_comment + category columns."""
        col_set = set(df_columns)
        return REQUIRED_COLUMNS.issubset(col_set)

    def _transform_row(self, row: dict[str, Any]) -> Document:
        """Transform a Reddit row into a Document."""
        text = self._safe_str(row.get("clean_comment"))
        if not text:
            raise ValueError("Empty comment text")

        category = self._safe_str(row.get("category"))

        metadata: dict[str, Any] = {}
        if category is not None:
            metadata["sentiment_category"] = category

        return Document(
            platform=self.platform,
            source_dataset="Reddit_Data.csv",
            text=text,
            metadata=metadata,
            raw_record={k: str(v) for k, v in row.items() if v is not None},
        )
