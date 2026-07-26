"""Tests for storage layers (JSONL, Parquet, Manifest)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.ingestion.models import RawDocument, SourceType
from pain_intelligence.ingestion.storage.jsonl_store import JsonlStore
from pain_intelligence.ingestion.storage.manifest import ManifestBuilder
from pain_intelligence.ingestion.storage.parquet_store import ParquetStore


def _make_doc(doc_id: str = "abc123", source: SourceType = SourceType.GITHUB) -> RawDocument:
    return RawDocument(
        document_id=doc_id,
        source=source,
        source_type="issue",
        external_id="ext1",
        title="Test Issue",
        content="Body text here",
        author="user1",
        tags=["bug"],
    )


class TestJsonlStore:
    def test_write_and_read(self, tmp_dir: Path):
        store = JsonlStore(tmp_dir)
        docs = [_make_doc("doc1"), _make_doc("doc2")]

        path = store.write("github", docs)
        assert path.exists()
        assert path.suffix == ".jsonl"

        # Read back
        records = store.read("github")
        assert len(records) == 2
        assert records[0]["document_id"] == "doc1"

    def test_append_mode(self, tmp_dir: Path):
        store = JsonlStore(tmp_dir)
        store.write("github", [_make_doc("doc1")])
        store.write("github", [_make_doc("doc2")])

        records = store.read("github")
        assert len(records) == 2

    def test_read_nonexistent(self, tmp_dir: Path):
        store = JsonlStore(tmp_dir)
        records = store.read("nonexistent_source")
        assert records == []


class TestParquetStore:
    def test_write_and_read(self, tmp_dir: Path):
        store = ParquetStore(tmp_dir)
        docs = [_make_doc("doc1"), _make_doc("doc2")]

        path = store.write("github", docs)
        assert path.exists()
        assert path.suffix == ".parquet"

        # Read back as DataFrame
        df = store.read("github")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_parquet_schema(self, tmp_dir: Path):
        store = ParquetStore(tmp_dir)
        docs = [_make_doc("doc1")]
        store.write("github", docs)

        df = store.read("github")
        assert "document_id" in df.columns
        assert "source" in df.columns
        assert "title" in df.columns
        assert "content" in df.columns

    def test_read_nonexistent(self, tmp_dir: Path):
        store = ParquetStore(tmp_dir)
        df = store.read("nonexistent")
        assert df.is_empty()


class TestManifestBuilder:
    def test_write_manifest(self, tmp_dir: Path):
        builder = ManifestBuilder(tmp_dir)
        docs = [_make_doc("doc1"), _make_doc("doc2")]

        path = builder.write("github", docs, file_paths=["file1.jsonl", "file1.parquet"])
        assert path.exists()
        assert path.name == "github_manifest.json"

        # Read and verify
        import json
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["document_count"] == 2
        assert manifest["source"] == "github"
        assert len(manifest["file_paths"]) == 2
