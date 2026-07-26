"""ClusteringEngine — orchestrates the semantic clustering pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.knowledge.manifest import PipelineManifest, generate_run_id
from pain_intelligence.knowledge.metadata import get_run_id_from_asset, read_parquet_metadata
from phase2.clustering.builder import ClusterBuilder
from phase2.clustering.config import ClusteringConfig, load_clustering_config
from phase2.clustering.evaluator import ClusterEvaluator
from phase2.clustering.exporter import (
    build_cluster_report,
    generate_text_report,
    write_json_report,
    write_manifest,
)
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.metrics import compute_cluster_stats
from phase2.clustering.providers import create_provider
from phase2.clustering.schema import ClusterManifest, SemanticCluster
from phase2.clustering.search import ClusterSearcher
from phase2.clustering.store import SemanticClusterStore
from phase2.clustering.validator import ClusterValidator


def _load_relationship_graph(
    relationship_path: Path,
    relationship_threshold: float,
) -> RelationshipGraph:
    if not relationship_path.exists():
        return RelationshipGraph()

    df = pl.read_parquet(str(relationship_path))
    if df.height == 0:
        return RelationshipGraph()

    edges: list[RelationshipEdge] = []
    for row in df.iter_rows(named=True):
        sim = row.get("similarity_score", 0.0)
        if sim < relationship_threshold:
            continue
        edges.append(
            RelationshipEdge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                similarity=sim,
                confidence=row.get("confidence", 0.0),
                relationship_type=row.get("relationship_type", "similar"),
            )
        )

    return RelationshipGraph.from_edges(edges)


def _load_relationship_manifest_hash(output_directory: Path) -> str:
    path = output_directory / "similarity_manifest.json"
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


class ClusteringEngine:
    """High-level orchestrator for the semantic clustering pipeline.

    Flow:
        Relationships -> RelationshipGraph -> ClusterProvider -> ClusterBuilder
        -> ClusterEvaluator -> ClusterValidator -> SemanticClusterStore -> Exporter

    Validates input relationship assets before processing.
    ALWAYS overwrites previous cluster outputs.
    """

    def __init__(self, config: ClusteringConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or generate_run_id()
        self._provider = create_provider(config)
        self._store = SemanticClusterStore(config.output_directory)
        self._builder = ClusterBuilder(config, self._provider)
        self._evaluator = ClusterEvaluator(config)
        self._validator = ClusterValidator(config)
        self._manifest = PipelineManifest(
            config.output_directory.parent.parent if config.output_directory.parent.parent.exists() else Path("pain_intelligence/knowledge"),
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    def _validate_input_relationships(self) -> Path | None:
        """Validate that the relationship asset exists and run_id is consistent."""
        relationship_path = self._config.output_directory / "semantic_relationships.parquet"

        if not relationship_path.exists():
            return None

        # Check run_id consistency
        asset_run_id = get_run_id_from_asset(relationship_path)
        manifest_entry = self._manifest.get_asset("semantic_relationships.parquet")
        manifest_run_id = manifest_entry.get("run_id", "") if manifest_entry else ""

        if asset_run_id and manifest_run_id and asset_run_id != manifest_run_id:
            return None

        return relationship_path

    def generate(self, force: bool = False) -> dict[str, Any]:
        """Run the full cluster generation pipeline.

        ALWAYS overwrites previous cluster outputs.
        """
        start = time.perf_counter()

        # Validate input relationships
        relationship_path = self._validate_input_relationships()
        if relationship_path is None:
            relationship_path = self._config.output_directory / "semantic_relationships.parquet"

        graph = _load_relationship_graph(relationship_path, self._config.relationship_threshold)

        if len(graph) == 0:
            self._store.save([])
            elapsed = time.perf_counter() - start
            return {
                "total_clusters": 0,
                "status": "completed",
                "reason": "no relationships found or graph is empty",
                "elapsed_seconds": round(elapsed, 2),
            }

        raw_clusters = self._provider.cluster(graph, self._config)

        if self._config.remove_singletons:
            raw_clusters = [c for c in raw_clusters if len(c) > 1]

        raw_clusters = [c for c in raw_clusters if len(c) <= self._config.maximum_cluster_size]

        if self._config.merge_small_clusters:
            raw_clusters = self._merge_small_clusters(raw_clusters, graph)

        raw_clusters = [c for c in raw_clusters if len(c) >= self._config.minimum_cluster_size]

        clusters: list[SemanticCluster] = []
        for members in raw_clusters:
            cluster = self._builder.build(members, graph)
            clusters.append(cluster)

        clusters = [self._evaluator.evaluate_and_update(c, graph) for c in clusters]

        validation = self._validator.validate(clusters, graph)

        # ALWAYS store, even if empty
        self._store.save(clusters)

        elapsed = time.perf_counter() - start

        stats = compute_cluster_stats(clusters)

        total_relationships = sum(c.relationship_count for c in clusters)

        all_members = set()
        for c in clusters:
            all_members.update(c.member_ids)
        orphan_count = len(graph.nodes()) - len(all_members)

        rel_manifest_hash = _load_relationship_manifest_hash(self._config.output_directory)
        manifest = ClusterManifest(
            provider=self._provider.name,
            provider_version=self._provider.version,
            algorithm=self._provider.name,
            record_count=stats.total_clusters,
            member_count=stats.total_members,
            relationship_count=total_relationships,
            generated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(elapsed, 2),
            config_hash=self._config.config_hash,
            relationship_manifest_hash=rel_manifest_hash,
        )
        write_manifest(manifest, self._config.output_directory)

        report = build_cluster_report(
            clusters=clusters,
            elapsed_seconds=elapsed,
            provider=self._provider.name,
            algorithm=self._provider.name,
            relationship_count=total_relationships,
            orphan_concept_count=orphan_count,
        )
        write_json_report(report, self._config.output_directory)
        generate_text_report(report, self._config.output_directory)

        # Update pipeline manifest
        self._manifest.start_run()
        cluster_path = self._config.output_directory / "semantic_clusters.parquet"
        self._manifest.register_asset(
            name="semantic_clusters.parquet",
            path=cluster_path,
            record_count=stats.total_clusters,
            stage="clustering",
        )
        self._manifest.complete_run()
        self._manifest.save()

        return {
            "total_clusters": stats.total_clusters,
            "total_members": stats.total_members,
            "total_relationships": total_relationships,
            "average_cluster_size": round(stats.average_cluster_size, 2),
            "average_density": round(stats.average_density, 6),
            "average_quality": round(stats.average_quality, 6),
            "low_quality_clusters": report.low_quality_count,
            "orphan_concepts": orphan_count,
            "singletons": report.singleton_count,
            "provider": self._provider.name,
            "algorithm": self._provider.name,
            "valid": validation.valid,
            "validation_issues": len(validation.issues),
            "elapsed_seconds": round(elapsed, 2),
        }

    def _merge_small_clusters(
        self,
        clusters: list[list[str]],
        graph: RelationshipGraph,
    ) -> list[list[str]]:
        small = [c for c in clusters if len(c) < self._config.minimum_cluster_size]
        large = [c for c in clusters if len(c) >= self._config.minimum_cluster_size]

        for small_cluster in small:
            best_target_idx = -1
            best_similarity = -1.0

            for i, large_cluster in enumerate(large):
                total_sim = 0.0
                count = 0
                for u in small_cluster:
                    for v in large_cluster:
                        w = graph.edge_weight(u, v)
                        if w is not None:
                            total_sim += w
                            count += 1
                avg_sim = total_sim / max(count, 1)

                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_target_idx = i

            if best_target_idx >= 0:
                merged = sorted(set(large[best_target_idx] + small_cluster))
                if len(merged) <= self._config.maximum_cluster_size:
                    large[best_target_idx] = merged
                else:
                    large.append(sorted(small_cluster))
            else:
                large.append(sorted(small_cluster))

        return large

    def stats(self) -> dict[str, Any]:
        clusters = self._store.load()
        stats_val = compute_cluster_stats(clusters)
        return stats_val.model_dump()

    def search_clusters(self, query_id: str) -> list[SemanticCluster]:
        clusters = self._store.load()
        searcher = ClusterSearcher(clusters)

        result = searcher.find_cluster(query_id)
        if result is not None:
            return [result]

        result = searcher.find_by_representative(query_id)
        if result is not None:
            return [result]

        return searcher.find_by_member(query_id)

    def verify(self) -> dict[str, Any]:
        """Verify integrity of stored clusters with run_id validation."""
        clusters = self._store.load()
        relationship_path = self._config.output_directory / "semantic_relationships.parquet"
        graph = _load_relationship_graph(relationship_path, self._config.relationship_threshold)

        validation = self._validator.validate(clusters, graph)

        result: dict[str, Any] = {
            "valid": validation.valid,
            "clusters_checked": validation.clusters_checked,
            "members_checked": validation.members_checked,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "cluster_id": issue.cluster_id,
                    "member_id": issue.member_id,
                }
                for issue in validation.issues
            ],
            "total_clusters": len(clusters),
        }

        # Check run_id
        cluster_path = self._config.output_directory / "semantic_clusters.parquet"
        if cluster_path.exists():
            asset_run_id = get_run_id_from_asset(cluster_path)
            result["run_id"] = asset_run_id or "not found"
        else:
            result["run_id"] = "not found"

        return result
