"""Tests for schema models."""

from __future__ import annotations

from phase2.clustering.schema import (
    ClusterManifest,
    ClusterMember,
    ClusterMetrics,
    ClusterReport,
    ClusterStats,
    ClusterSummary,
    ClusterType,
    SemanticCluster,
    ValidationIssue,
    ValidationResult,
)


class TestSemanticCluster:
    def test_frozen_model(self) -> None:
        cluster = SemanticCluster(
            cluster_id="abc123",
            representative_id="rep1",
            member_ids=("mem1", "mem2", "mem3"),
            member_count=3,
            relationship_count=3,
            average_similarity=0.9,
            density=1.0,
            quality_score=0.85,
            cluster_type=ClusterType.NORMAL,
            provider="connected_components",
            provider_version="1.0",
            algorithm="connected_components",
            version="1.0",
        )
        assert cluster.cluster_id == "abc123"
        assert cluster.member_count == 3
        try:
            cluster.cluster_id = "new_id"
            assert False, "Should have raised"
        except Exception:
            pass

    def test_forbids_extra_fields(self) -> None:
        try:
            SemanticCluster(
                cluster_id="abc",
                representative_id="rep1",
                member_ids=("m1",),
                member_count=1,
                relationship_count=0,
                average_similarity=0.0,
                density=0.0,
                quality_score=0.0,
                provider="cc",
                provider_version="1.0",
                algorithm="cc",
                version="1.0",
                unknown_field="bad",
            )
            assert False, "Should have raised"
        except Exception:
            pass

    def test_member_ids_are_sorted(self) -> None:
        cluster = SemanticCluster(
            cluster_id="abc",
            representative_id="rep1",
            member_ids=("a", "b", "c", "d"),
            member_count=4,
            relationship_count=6,
            average_similarity=0.9,
            density=1.0,
            quality_score=0.9,
            provider="cc",
            provider_version="1.0",
            algorithm="cc",
            version="1.0",
        )
        assert list(cluster.member_ids) == sorted(cluster.member_ids)


class TestClusterType:
    def test_enum_values(self) -> None:
        assert ClusterType.NORMAL.value == "normal"
        assert ClusterType.LOW_QUALITY.value == "low_quality"


class TestValidationIssue:
    def test_frozen(self) -> None:
        issue = ValidationIssue(
            severity="ERROR",
            code="MIN_SIZE",
            message="Too small",
        )
        assert issue.severity == "ERROR"
        try:
            issue.severity = "WARN"
            assert False
        except Exception:
            pass


class TestValidationResult:
    def test_valid_by_default(self) -> None:
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.issues == []


class TestClusterManifest:
    def test_manifest_fields(self) -> None:
        m = ClusterManifest(
            provider="connected_components",
            provider_version="1.0",
            algorithm="connected_components",
            record_count=10,
            member_count=50,
            relationship_count=100,
            generated_at="2026-01-01T00:00:00",
            elapsed_seconds=1.0,
            config_hash="abc123",
            relationship_manifest_hash="def456",
        )
        assert m.project == "pain-intelligence-engine"
        assert m.module == "clustering"
        assert m.record_count == 10


class TestClusterReport:
    def test_report_fields(self) -> None:
        report = ClusterReport(
            generated_at="2026-01-01T00:00:00",
            elapsed_seconds=1.0,
            total_clusters=5,
            total_members=25,
            total_relationships=50,
            average_cluster_size=5.0,
            cluster_density=0.5,
            low_quality_count=1,
            orphan_concept_count=0,
            singleton_count=0,
            provider="cc",
            algorithm="cc",
        )
        assert report.total_clusters == 5
        assert report.report_type == "cluster_report"
