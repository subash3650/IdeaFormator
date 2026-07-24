"""Statistics computation for processed datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl

from pain_intelligence.schema.document import Document


def compute_statistics(
    documents: list[Document],
    removed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive statistics for a list of Documents.

    Args:
        documents: List of processed Document objects.
        removed: List of removed document records.

    Returns:
        Dictionary of statistics.
    """
    total = len(documents)

    if total == 0:
        return _empty_statistics()

    platform_counter: Counter[str] = Counter()
    rating_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    missing_values: dict[str, int] = {
        "title": 0,
        "rating": 0,
        "author": 0,
        "country": 0,
        "language": 0,
        "created_at": 0,
    }
    lengths: list[int] = []
    ratings: list[float] = []

    for doc in documents:
        platform_counter[doc.platform.value] += 1
        lengths.append(doc.document_length)

        if doc.title is None:
            missing_values["title"] += 1
        if doc.rating is None:
            missing_values["rating"] += 1
        else:
            rating_counter[str(int(doc.rating))] += 1
            ratings.append(doc.rating)
        if doc.author is None:
            missing_values["author"] += 1
        if doc.country is None:
            missing_values["country"] += 1
        else:
            country_counter[doc.country] += 1
        if doc.language is None:
            missing_values["language"] += 1
        else:
            language_counter[doc.language] += 1
        if doc.created_at is None:
            missing_values["created_at"] += 1

    avg_length = sum(lengths) / total if lengths else 0.0
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

    stats: dict[str, Any] = {
        "total_documents": total,
        "platform_distribution": dict(platform_counter.most_common()),
        "rating_distribution": dict(rating_counter.most_common()),
        "language_distribution": dict(language_counter.most_common()),
        "country_distribution": dict(country_counter.most_common()),
        "average_document_length": round(avg_length, 2),
        "average_rating": round(avg_rating, 2),
        "missing_values": missing_values,
        "missing_value_percentages": {
            k: round(v / total * 100, 2) for k, v in missing_values.items()
        },
    }

    if removed:
        removal_reasons: Counter[str] = Counter()
        for r in removed:
            removal_reasons[r.get("reason", "unknown")] += 1
        stats["removed_documents"] = len(removed)
        stats["removal_reasons"] = dict(removal_reasons.most_common())
    else:
        stats["removed_documents"] = 0
        stats["removal_reasons"] = {}

    return stats


def _empty_statistics() -> dict[str, Any]:
    """Return an empty statistics dictionary."""
    return {
        "total_documents": 0,
        "platform_distribution": {},
        "rating_distribution": {},
        "language_distribution": {},
        "country_distribution": {},
        "average_document_length": 0,
        "average_rating": 0,
        "missing_values": {},
        "missing_value_percentages": {},
        "removed_documents": 0,
        "removal_reasons": {},
    }
