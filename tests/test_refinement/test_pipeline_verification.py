"""Tests for pipeline verification and dashboard (Issues 4, 5, 6)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.manifest import PIPELINE_VERSION, SCHEMA_VERSION, PipelineManifest, compute_checksum, generate_run_id
from pain_intelligence.knowledge.metadata import make_asset_metadata, write_parquet_with_metadata
from pain_intelligence.knowledge.store import KnowledgeStore
from pain_intelligence.pipeline.verify import generate_dashboard, verify_pipeline


class TestPipelineVerification:
    """Verify the pipeline verify command."""

    def test_verify_pipeline_returns_report(self):
        """verify_pipeline returns a structured report."""
        report = verify_pipeline(fix=False)

        assert "overall" in report
        assert "checks" in report
        assert "timestamp" in report
        assert "pipeline_version" in report
        assert report["pipeline_version"] == PIPELINE_VERSION

    def test_verify_checks_format(self):
        """Each check has status and detail."""
        report = verify_pipeline(fix=False)

        for check_name, check_result in report.get("checks", {}).items():
            assert "status" in check_result, f"Check {check_name} missing status"
            assert check_result["status"] in ("PASS", "FAIL", "WARN", "SKIP"), f"Unexpected status in {check_name}"
            assert "detail" in check_result, f"Check {check_name} missing detail"

    def test_verify_asset_checks(self):
        """Asset checks are performed."""
        report = verify_pipeline(fix=False)

        asset_checks = [k for k in report["checks"] if k.startswith("asset:")]
        assert len(asset_checks) > 0
        # At minimum check knowledge assets
        assert any("observations" in c for c in asset_checks)

    def test_verify_summary_counts(self):
        """Summary has total/passed/failed/warned counts."""
        report = verify_pipeline(fix=False)

        summary = report.get("summary", {})
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "warned" in summary
        assert summary["total_checks"] == summary["passed"] + summary["failed"] + summary["warned"]


class TestRunIdPropagation:
    """Verify run_id is propagated through assets."""

    def test_generate_run_id_format(self):
        """Run ID follows YYYYMMDD-HHMMSS-ffffff format."""
        run_id = generate_run_id()
        parts = run_id.split("-")
        assert len(parts) == 3
        assert parts[0].isdigit()
        assert parts[1].isdigit()
        assert parts[2].isdigit()

    def test_manifest_run_id_persists(self, tmp_path: Path):
        """Run ID survives manifest save/load cycle."""
        manifest = PipelineManifest(str(tmp_path))
        run_id = manifest.start_run()
        manifest.save()

        manifest2 = PipelineManifest(str(tmp_path))
        assert manifest2.run_id == run_id

    def test_asset_metadata_contains_run_id(self, tmp_path: Path):
        """Asset metadata contains run_id, generated_at, pipeline_version."""
        df = pl.DataFrame({"x": [1, 2]})
        meta = make_asset_metadata(
            run_id="test-run-42",
            input_checksum="chk123",
            input_document_count=100,
        )
        path = tmp_path / "test.parquet"
        write_parquet_with_metadata(df, path, metadata=meta)

        from pain_intelligence.knowledge.metadata import read_parquet_metadata
        loaded = read_parquet_metadata(path)

        assert loaded["run_id"] == "test-run-42"
        assert loaded["pipeline_version"] == PIPELINE_VERSION
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["input_checksum"] == "chk123"
        assert loaded["input_document_count"] == "100"
        assert "generated_at" in loaded

    def test_checksum_computation(self, tmp_path: Path):
        """compute_checksum returns consistent SHA-256 prefixes."""
        path = tmp_path / "data.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        df.write_parquet(str(path))

        cs1 = compute_checksum(path)
        cs2 = compute_checksum(path)
        assert cs1 == cs2
        assert len(cs1) == 16

        # Different data produces different checksum
        path2 = tmp_path / "data2.parquet"
        df2 = pl.DataFrame({"x": [4, 5, 6]})
        df2.write_parquet(str(path2))
        cs3 = compute_checksum(path2)
        assert cs1 != cs3


class TestChecksumPropagation:
    """Verify checksums are computed and stored correctly."""

    def test_store_records_checksum_in_metadata(self, tmp_path: Path):
        """KnowledgeStore.write_asset includes checksum in Parquet metadata."""
        store = KnowledgeStore(str(tmp_path / "knowledge"))
        store.run_id = "checksum-test"

        df = pl.DataFrame({"observation_id": ["1"], "type": ["entity"], "value": ["checksum_test"],
                           "document_id": ["d1"], "platform": ["web"], "rating": [3.0],
                           "country": ["US"], "text_snippet": [""], "extractor": ["test"],
                           "method": ["heuristic"], "confidence": [0.5],
                           "entity": [None], "entity_type": [None], "category": [None],
                           "pattern_label": [None], "canonical_value": [None],
                           "canonical_source": [None], "pipeline_version": [""], "generated_at": [""]})
        store.write_asset("observations", df, input_checksum="abc123", input_document_count=50)

        meta = store.read_asset_metadata("observations")
        assert meta.get("input_checksum") == "abc123"
        assert meta.get("input_document_count") == "50"


class TestDashboardGeneration:
    """Verify dashboard generation."""

    def test_dashboard_returns_dict(self, tmp_path: Path):
        """generate_dashboard returns a dict with expected keys."""
        dash = generate_dashboard(output_dir=str(tmp_path))

        assert "generated_at" in dash
        assert "pipeline_version" in dash
        assert "documents" in dash
        assert "observation_count" in dash
        assert "evidence_count" in dash
        assert "problem_signal_count" in dash
        assert "embedding_count" in dash
        assert "relationship_count" in dash
        assert "cluster_count" in dash
        assert "overall_status" in dash

    def test_dashboard_writes_files(self, tmp_path: Path):
        """Dashboard writes both JSON and TXT files."""
        generate_dashboard(output_dir=str(tmp_path))

        json_path = tmp_path / "pipeline_dashboard.json"
        txt_path = tmp_path / "pipeline_dashboard.txt"

        assert json_path.exists()
        assert txt_path.exists()

        import json
        with open(json_path) as f:
            data = json.load(f)
        assert "overall_status" in data

    def test_dashboard_txt_format(self, tmp_path: Path):
        """Text dashboard contains section headers."""
        generate_dashboard(output_dir=str(tmp_path))

        txt_path = tmp_path / "pipeline_dashboard.txt"
        content = txt_path.read_text(encoding="utf-8")

        assert "PIPELINE DASHBOARD" in content
        assert "DOCUMENTS" in content
        assert "KNOWLEDGE EXTRACTION" in content
        assert "EMBEDDINGS" in content
        assert "CLUSTERS" in content
