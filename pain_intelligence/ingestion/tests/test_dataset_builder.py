"""Integration tests for DatasetBuilder compatibility adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pain_intelligence.ingestion.dataset_builder import DatasetBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_source_parquet(path: Path, records: list[dict]) -> None:
    """Write a minimal ingestion-format Parquet file using pyarrow."""
    schema = pa.schema([
        ("document_id", pa.string()),
        ("external_id", pa.string()),
        ("source", pa.string()),
        ("content", pa.string()),
        ("clean_text", pa.string()),
        ("title", pa.string()),
        ("language", pa.string()),
        ("created_at", pa.string()),
        ("collected_at", pa.string()),
        ("url", pa.string()),
        ("author", pa.string()),
        ("metadata", pa.string()),
        ("version", pa.string()),
        ("schema_version", pa.string()),
    ])

    table = pa.table(
        {f.name: [r.get(f.name, "") for r in records] for f in schema},
        schema=schema,
    )
    pq.write_table(table, path)


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    """Create a knowledge base with normalized/{source}/ subdirs."""
    base = tmp_path / "knowledge"
    normalized = base / "normalized"
    normalized.mkdir(parents=True)
    processed = base / "processed"
    processed.mkdir()
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScanParquetFiles:
    def test_finds_all_sources(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        for src in ("github", "youtube", "playstore"):
            src_dir = norm / src
            src_dir.mkdir()
            _write_source_parquet(src_dir / f"{src}.parquet", [
                {"document_id": f"{src}-1", "source": src, "external_id": "1",
                 "content": "hello", "clean_text": "hello"},
            ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        files = builder._scan_parquet_files()

        assert len(files) == 3
        assert sorted(files.keys()) == ["github", "playstore", "youtube"]

    def test_picks_latest_when_multiple(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        src_dir = norm / "github"
        src_dir.mkdir()
        _write_source_parquet(src_dir / "2026-01-01.parquet", [
            {"document_id": "old", "source": "github", "external_id": "1",
             "content": "old", "clean_text": "old"},
        ])
        _write_source_parquet(src_dir / "2026-06-01.parquet", [
            {"document_id": "new", "source": "github", "external_id": "2",
             "content": "new", "clean_text": "new"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        files = builder._scan_parquet_files()

        assert len(files) == 1
        assert files["github"].name == "2026-06-01.parquet"

    def test_ignores_non_directories(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        _write_source_parquet(norm / "stray.parquet", [
            {"document_id": "s1", "source": "stray", "external_id": "1",
             "content": "x", "clean_text": "x"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        files = builder._scan_parquet_files()
        assert len(files) == 0

    def test_empty_when_no_files(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        assert builder._scan_parquet_files() == {}


class TestMapSchema:
    def test_maps_raw_document_to_legacy(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        df = pl.DataFrame({
            "document_id": ["abc123"],
            "external_id": ["GH-1"],
            "source": ["github"],
            "content": ["This is a GitHub issue"],
            "clean_text": ["this is a github issue"],
            "title": ["Bug report"],
            "language": ["en"],
            "created_at": ["2026-01-15T10:00:00Z"],
            "collected_at": ["2026-06-01T12:00:00Z"],
            "url": ["https://github.com/example/issue/1"],
            "author": ["testuser"],
            "metadata": ["{}"],
            "version": ["1"],
            "schema_version": ["1"],
        })

        mapped = builder._map_schema(df, "github")

        assert mapped.height == 1
        row = {col: mapped[col][0] for col in mapped.columns}

        assert row["id"] == "abc123"
        assert row["platform"] == "github"
        assert row["source_dataset"] == "github"
        assert row["text"] == "This is a GitHub issue"
        assert row["clean_text"] == "This is a GitHub issue"
        assert row["title"] == "Bug report"
        assert row["language"] == "en"
        assert row["author"] == "testuser"
        assert row["location"] == ""
        assert row["document_length"] > 0
        assert row["rating"] is None

    def test_maps_playstore_with_rating_and_country(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        df = pl.DataFrame({
            "document_id": ["ps1"],
            "external_id": ["PS-1"],
            "source": ["playstore"],
            "content": ["Great app!"],
            "clean_text": ["great app"],
            "title": ["Review"],
            "language": ["en"],
            "created_at": ["2026-03-10T08:00:00Z"],
            "collected_at": ["2026-06-01T12:00:00Z"],
            "url": ["https://play.google.com/store/apps/details?id=com.test"],
            "author": ["reviewer"],
            "metadata": ['{"star_rating": 5, "country": "US"}'],
            "version": ["1"],
            "schema_version": ["1"],
        })

        mapped = builder._map_schema(df, "playstore")
        row = {col: mapped[col][0] for col in mapped.columns}

        assert row["rating"] == 5.0
        assert row["country"] == "US"
        assert row["platform"] == "playstore"

    def test_handles_missing_columns_gracefully(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        df = pl.DataFrame({
            "document_id": ["minimal"],
            "content": ["some text"],
        })

        mapped = builder._map_schema(df, "github")
        assert mapped.height == 1
        row = {col: mapped[col][0] for col in mapped.columns}
        assert row["id"] == "minimal"
        assert row["platform"] == "github"
        assert row["title"] == ""
        assert row["author"] == ""


class TestDeduplicate:
    def test_removes_duplicates_by_id(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        df = pl.DataFrame({
            "id": ["same", "same", "unique"],
            "text": ["v1", "v2", "u1"],
        })

        deduped = builder._deduplicate(df)
        assert deduped.height == 2

    def test_empty_dataframe(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        df = pl.DataFrame({"id": pl.Series([], dtype=pl.Utf8), "text": pl.Series([], dtype=pl.Utf8)})
        result = builder._deduplicate(df)
        assert result.is_empty()


class TestComputeStats:
    def test_returns_source_breakdown(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        source_dfs = {
            "github": pl.DataFrame({
                "id": ["g1", "g2"],
                "created_at": ["2026-01-15", "2026-02-20"],
            }),
            "youtube": pl.DataFrame({
                "id": ["y1"],
                "created_at": ["2026-03-10"],
            }),
        }

        start = datetime.now(timezone.utc)
        stats = builder._compute_stats(source_dfs, duplicates_removed=0, start_time=start)

        assert stats["total_documents"] == 3
        assert stats["duplicates_removed"] == 0
        assert stats["final_documents"] == 3
        assert stats["sources"]["github"]["documents"] == 2
        assert stats["sources"]["youtube"]["documents"] == 1
        assert stats["sources"]["github"]["first_seen"] == "2026-01-15"


class TestBuild:
    def test_full_build(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        github_dir = norm / "github"
        github_dir.mkdir()
        _write_source_parquet(github_dir / "batch.parquet", [
            {"document_id": "g1", "source": "github", "external_id": "1",
             "content": "issue text", "clean_text": "issue text", "title": "Issue",
             "language": "en", "created_at": "2026-01-15T10:00:00Z",
             "collected_at": "2026-06-01T12:00:00Z", "url": "https://example.com",
             "author": "dev", "metadata": "{}", "version": "1", "schema_version": "1"},
            {"document_id": "g2", "source": "github", "external_id": "2",
             "content": "another issue", "clean_text": "another issue", "title": "Issue 2",
             "language": "en", "created_at": "2026-02-20T10:00:00Z",
             "collected_at": "2026-06-01T12:00:00Z", "url": "https://example.com/2",
             "author": "dev", "metadata": "{}", "version": "1", "schema_version": "1"},
        ])

        ps_dir = norm / "playstore"
        ps_dir.mkdir()
        _write_source_parquet(ps_dir / "batch.parquet", [
            {"document_id": "p1", "source": "playstore", "external_id": "3",
             "content": "great app", "clean_text": "great app", "title": "Review",
             "language": "en", "created_at": "2026-03-10T08:00:00Z",
             "collected_at": "2026-06-01T12:00:00Z",
             "url": "https://play.google.com/store/apps/details?id=com.test",
             "author": "reviewer", "metadata": '{"star_rating": 5, "country": "US"}',
             "version": "1", "schema_version": "1"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        result = builder.build()

        assert result["status"] == "success"
        assert result["total_documents"] == 3
        assert result["final_documents"] == 3
        assert result["duplicates_removed"] == 0
        assert len(result["sources"]) == 2

        output_path = Path(result["output_path"])
        assert output_path.exists()

        df = pl.read_parquet(output_path)
        assert df.height == 3
        assert "id" in df.columns
        assert "platform" in df.columns

    def test_build_with_cross_source_duplicates(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        for src in ("github", "playstore"):
            src_dir = norm / src
            src_dir.mkdir()
            _write_source_parquet(src_dir / "batch.parquet", [
                {"document_id": "shared-id", "source": src, "external_id": "1",
                 "content": "text", "clean_text": "text"},
            ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        result = builder.build()

        assert result["total_documents"] == 2
        assert result["duplicates_removed"] == 1
        assert result["final_documents"] == 1

    def test_build_empty_returns_status(self, kb_dir: Path) -> None:
        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        result = builder.build()

        assert result["status"] == "empty"
        assert result["total_documents"] == 0

    def test_build_skips_when_output_exists(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        github_dir = norm / "github"
        github_dir.mkdir()
        _write_source_parquet(github_dir / "batch.parquet", [
            {"document_id": "g1", "source": "github", "external_id": "1",
             "content": "text", "clean_text": "text"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))

        result1 = builder.build()
        assert result1["status"] == "success"

        result2 = builder.build()
        assert result2["status"] == "skipped"

    def test_build_force_overwrites(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        github_dir = norm / "github"
        github_dir.mkdir()
        _write_source_parquet(github_dir / "batch.parquet", [
            {"document_id": "g1", "source": "github", "external_id": "1",
             "content": "text", "clean_text": "text"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        builder.build()

        result = builder.build(force=True)
        assert result["status"] == "success"
        assert result["final_documents"] == 1

    def test_report_json_written(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        github_dir = norm / "github"
        github_dir.mkdir()
        _write_source_parquet(github_dir / "batch.parquet", [
            {"document_id": "g1", "source": "github", "external_id": "1",
             "content": "text", "clean_text": "text"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        result = builder.build()

        report_path = Path(result["report_path"])
        assert report_path.exists()

        import json
        with open(report_path) as f:
            report = json.load(f)
        assert report["total_documents"] == 1
        assert "adapter_note" in report

    def test_output_has_all_legacy_columns(self, kb_dir: Path) -> None:
        norm = kb_dir / "normalized"
        github_dir = norm / "github"
        github_dir.mkdir()
        _write_source_parquet(github_dir / "batch.parquet", [
            {"document_id": "g1", "source": "github", "external_id": "1",
             "content": "text", "clean_text": "text", "title": "Title",
             "language": "en", "created_at": "2026-01-15T10:00:00Z",
             "collected_at": "2026-06-01T12:00:00Z", "url": "https://example.com",
             "author": "dev", "metadata": "{}", "version": "1", "schema_version": "1"},
        ])

        builder = DatasetBuilder(knowledge_base=str(kb_dir))
        result = builder.build()

        df = pl.read_parquet(result["output_path"])
        expected_cols = [
            "id", "platform", "source_dataset", "title", "text", "clean_text",
            "rating", "author", "country", "location", "language", "created_at",
            "metadata", "raw_record", "document_length",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
