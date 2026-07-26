"""Regression tests for schema consistency with Polars."""
from datetime import datetime, timezone

import polars as pl
import pytest

from pain_intelligence.ingestion.models import RawDocument, SourceType


class TestPolarsNullInference:
    """Polars infers Null when first N rows are all-None.
    Verify to_flat_dict() normalizes None to empty string.
    """

    def test_updated_at_all_none_then_value(self) -> None:
        docs = []
        for i in range(120):
            doc = RawDocument(
                document_id=f"doc_{i}",
                source=SourceType.PLAYSTORE,
                source_type="review",
                external_id=f"ext_{i}",
                content=f"Review {i}",
                author=f"User{i}",
                created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                updated_at=None,
                metadata={"star_rating": 4},
                raw_json={},
                tags=["rating:4"],
                categories=[],
            )
            docs.append(doc)

        doc_with_reply = RawDocument(
            document_id="doc_200",
            source=SourceType.PLAYSTORE,
            source_type="review",
            external_id="ext_200",
            content="Review with reply",
            author="User200",
            created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            updated_at=datetime(2026, 7, 24, 17, 31, 31, tzinfo=timezone.utc),
            metadata={"star_rating": 5},
            raw_json={},
            tags=["rating:5"],
            categories=[],
        )
        docs.append(doc_with_reply)

        rows = [doc.to_flat_dict() for doc in docs]
        df = pl.DataFrame(rows)

        assert df["updated_at"].dtype == pl.String
        assert df["updated_at"][0] == ""
        assert df["updated_at"][120] == "2026-07-24T17:31:31+00:00"

    def test_title_all_none(self) -> None:
        docs = [
            RawDocument(
                document_id=f"doc_{i}",
                source=SourceType.PLAYSTORE,
                source_type="review",
                external_id=f"ext_{i}",
                content=f"Review {i}",
                author="User",
                created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                metadata={},
                raw_json={},
                tags=[],
                categories=[],
            )
            for i in range(120)
        ]

        rows = [doc.to_flat_dict() for doc in docs]
        df = pl.DataFrame(rows)

        assert df["title"].dtype == pl.String
        assert df["title"][0] == ""

    def test_mixed_timestamp_inputs(self) -> None:
        """Parse datetime, ISO string, and None through the full cycle."""
        from pain_intelligence.ingestion.utils import parse_timestamp

        values = [
            parse_timestamp(datetime(2026, 1, 15, tzinfo=timezone.utc)),
            parse_timestamp("2026-02-20T10:00:00+00:00"),
            parse_timestamp(None),
        ]

        docs = []
        for i, val in enumerate(values):
            docs.append(RawDocument(
                document_id=f"doc_{i}",
                source=SourceType.PLAYSTORE,
                source_type="review",
                external_id=f"ext_{i}",
                content=f"Review {i}",
                author="User",
                created_at=val,
                metadata={},
                raw_json={},
                tags=[],
                categories=[],
            ))

        rows = [doc.to_flat_dict() for doc in docs]
        df = pl.DataFrame(rows)

        assert df["created_at"].dtype == pl.String
        assert df["created_at"][0] == "2026-01-15T00:00:00+00:00"
        assert df["created_at"][1] == "2026-02-20T10:00:00+00:00"
        # None becomes empty string after normalization
        assert df["created_at"][2] == ""

    def test_to_flat_dict_never_returns_none(self) -> None:
        """All fields in to_flat_dict output should be str, never None."""
        doc = RawDocument(
            document_id="test",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="1",
            content="test",
            author="user",
            created_at=None,
            metadata={},
            raw_json={},
            tags=[],
            categories=[],
        )

        flat = doc.to_flat_dict()
        for field_name, value in flat.items():
            assert value is not None, f"Field {field_name} is None in to_flat_dict"
            assert isinstance(value, str), (
                f"Field {field_name} is {type(value).__name__}, expected str"
            )
