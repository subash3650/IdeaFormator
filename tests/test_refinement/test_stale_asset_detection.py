"""Tests for stale asset detection (Issue 1)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.exceptions import MissingAssetError, StaleAssetError
from pain_intelligence.knowledge.manifest import PIPELINE_VERSION, SCHEMA_VERSION, PipelineManifest
from pain_intelligence.knowledge.metadata import (
    get_run_id_from_asset,
    make_asset_metadata,
    read_parquet_metadata,
    write_parquet_with_metadata,
)
from pain_intelligence.knowledge.store import ASSETS, EMPTY_SCHEMAS, KnowledgeStore


class TestStaleAssetDetection:
    """Verify StaleAssetError is raised when run_ids don't match."""

    def test_stale_asset_error_message(self):
        """StaleAssetError produces a readable error message."""
        err = StaleAssetError(
            asset_path="test.parquet",
            expected_run_id="20260725-230501",
            actual_run_id="20260724-114200",
        )
        msg = str(err)
        assert "StaleAssetError" in msg
        assert "test.parquet" in msg
        assert "20260725-230501" in msg
        assert "20260724-114200" in msg

    def test_missing_asset_error(self):
        """MissingAssetError when file does not exist."""
        err = MissingAssetError("not_found.parquet")
        assert "MissingAssetError" in str(err)

    def test_metadata_roundtrip(self, tmp_path: Path):
        """Write Parquet with metadata and read it back."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        meta = make_asset_metadata(
            run_id="20260725-230501",
            input_checksum="abc123",
            input_document_count=100,
        )
        path = tmp_path / "test.parquet"
        write_parquet_with_metadata(df, path, metadata=meta)

        read_back = read_parquet_metadata(path)
        assert read_back["run_id"] == "20260725-230501"
        assert read_back["pipeline_version"] == PIPELINE_VERSION
        assert read_back["schema_version"] == SCHEMA_VERSION

        loaded = pl.read_parquet(str(path))
        assert loaded.height == 3

    def test_run_id_from_asset(self, tmp_path: Path):
        """get_run_id_from_asset extracts run_id correctly."""
        df = pl.DataFrame({"x": [1]})
        meta = make_asset_metadata(run_id="test-run-001")
        path = tmp_path / "test.parquet"
        write_parquet_with_metadata(df, path, metadata=meta)

        assert get_run_id_from_asset(path) == "test-run-001"
        assert get_run_id_from_asset(tmp_path / "nonexistent.parquet") == ""

    def test_store_read_asset_stale_detection(self, tmp_path: Path):
        """KnowledgeStore raises StaleAssetError when run_id doesn't match."""
        store = KnowledgeStore(str(tmp_path))
        store.run_id = "current-run"

        # Write an asset with the correct run_id
        df = pl.DataFrame({"observation_id": ["1"], "type": ["entity"], "value": ["test"],
                           "document_id": ["d1"], "platform": ["web"], "rating": [3.0],
                           "country": ["US"], "text_snippet": [""], "extractor": ["test"],
                           "method": ["heuristic"], "confidence": [0.5],
                           "entity": [None], "entity_type": [None], "category": [None],
                           "pattern_label": [None], "canonical_value": [None],
                           "canonical_source": [None], "pipeline_version": [""], "generated_at": [""]})
        store.write_asset("observations", df)

        # Read with matching run_id — should succeed
        result = store.read_asset("observations", validate_run_id=True)
        assert result.height == 1

        # Change run_id and try again — should raise StaleAssetError
        store.run_id = "different-run"
        with pytest.raises(StaleAssetError):
            store.read_asset("observations", validate_run_id=True)


class TestEmptyParquetOverwrite:
    """Verify empty parquet files are written with correct schema."""

    def test_store_writes_empty_parquet(self, tmp_path: Path):
        """KnowledgeStore.write_asset writes empty parquet with schema when no data."""
        store = KnowledgeStore(str(tmp_path))
        store.run_id = "test-run"

        empty_df = pl.DataFrame()
        path = store.write_asset("observations", empty_df)

        assert path.exists()
        # Should have the correct schema
        df = pl.read_parquet(str(path))
        expected_cols = set(EMPTY_SCHEMAS["observations"].keys())
        actual_cols = set(df.columns)
        assert expected_cols.issubset(actual_cols), f"Missing columns: {expected_cols - actual_cols}"
        assert df.height == 0

    def test_store_overwrites_previous_asset(self, tmp_path: Path):
        """Writing an asset twice overwrites the previous content."""
        store = KnowledgeStore(str(tmp_path))
        store.run_id = "test-run"

        # Write first version with data
        df1 = pl.DataFrame({"observation_id": ["1"], "type": ["entity"], "value": ["test"],
                           "document_id": ["d1"], "platform": ["web"], "rating": [3.0],
                           "country": ["US"], "text_snippet": [""], "extractor": ["test"],
                           "method": ["heuristic"], "confidence": [0.5],
                           "entity": [None], "entity_type": [None], "category": [None],
                           "pattern_label": [None], "canonical_value": [None],
                           "canonical_source": [None], "pipeline_version": [""], "generated_at": [""]})
        path1 = store.write_asset("observations", df1)
        assert pl.read_parquet(str(path1)).height == 1

        # Write second version — different data, should overwrite
        df2 = pl.DataFrame()
        path2 = store.write_asset("observations", df2)
        assert path1 == path2
        df_loaded = pl.read_parquet(str(path2))
        assert df_loaded.height == 0  # Should be empty now

    def test_empty_parquet_has_metadata(self, tmp_path: Path):
        """Empty parquet still contains embedded metadata."""
        store = KnowledgeStore(str(tmp_path))
        store.run_id = "empty-run-001"

        empty_df = pl.DataFrame()
        store.write_asset("evidence", empty_df)

        meta = store.read_asset_metadata("evidence")
        assert meta.get("run_id") == "empty-run-001"
        assert meta.get("pipeline_version") == PIPELINE_VERSION


class TestManifestValidation:
    """Verify manifest registration and validation."""

    def test_manifest_start_run(self, tmp_path: Path):
        """PipelineManifest.start_run generates a run_id."""
        manifest = PipelineManifest(str(tmp_path))
        run_id = manifest.start_run()
        assert run_id
        assert manifest.run_id == run_id

    def test_manifest_register_asset(self, tmp_path: Path):
        """Assets can be registered in the manifest."""
        manifest = PipelineManifest(str(tmp_path))
        run_id = manifest.start_run()

        # Create a dummy file
        asset_path = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1]})
        df.write_parquet(str(asset_path))

        manifest.register_asset(
            name="test.parquet",
            path=asset_path,
            record_count=1,
            stage="testing",
        )
        manifest.save()

        assert manifest.get_asset("test.parquet") is not None
        assert manifest.get_stage_status("testing") == "completed"

        # Verify persistence
        manifest2 = PipelineManifest(str(tmp_path))
        assert manifest2.get_asset("test.parquet") is not None
        assert manifest2.to_dict().get("run_id") == run_id

    def test_manifest_validate_upstream(self, tmp_path: Path):
        """validate_upstream raises StaleAssetError on run_id mismatch."""
        manifest = PipelineManifest(str(tmp_path))
        run_id = manifest.start_run()

        # Register an asset with current run_id
        asset_path = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1]})
        df.write_parquet(str(asset_path))

        manifest.register_asset(
            name="test.parquet",
            path=asset_path,
            record_count=1,
            stage="testing",
        )
        manifest.save()

        # Start a new run (different run_id)
        manifest.start_run()

        # Now validate_upstream should raise StaleAssetError
        with pytest.raises(StaleAssetError):
            manifest.validate_upstream("test.parquet", asset_path)

    def test_manifest_complete_run(self, tmp_path: Path):
        """complete_run sets completed_at and elapsed_seconds."""
        manifest = PipelineManifest(str(tmp_path))
        manifest.start_run()
        manifest.complete_run()
        manifest.save()

        data = manifest.to_dict()
        assert data.get("completed_at") is not None
