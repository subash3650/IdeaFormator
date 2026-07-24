"""SimilarityEngine – orchestrates the relationship generation pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from phase2.embeddings.schema import SourceType
from phase2.similarity.builder import RelationshipBuilder
from phase2.similarity.comparer import count_frequencies
from phase2.similarity.confidence import RelationshipConfidencePolicy
from phase2.similarity.config import SimilarityEngineConfig
from phase2.similarity.exporter import (
    generate_quality_report,
    write_json_report,
    write_manifest,
)
from phase2.similarity.filters import (
    DuplicateRelationshipFilter,
    FilterPipeline,
    RelationshipPolicyFilter,
    SelfSimilarityFilter,
    ThresholdFilter,
    TopKPerSourceFilter,
)
from phase2.similarity.indexes import LinearIndex
from phase2.similarity.metrics import RelationshipStats, compute_stats
from phase2.similarity.providers import cosine  # noqa: F401 – register
from phase2.similarity.providers import dot_product  # noqa: F401 – register
from phase2.similarity.providers import euclidean  # noqa: F401 – register
from phase2.similarity.providers.base import SimilarityProvider
from phase2.similarity.providers.registry import create_provider
from phase2.similarity.schema import RelationshipManifest, SemanticRelationship
from phase2.similarity.search import RelationshipSearcher
from phase2.similarity.statistics import (
    RelationshipStatistics,
    compute_relationship_statistics,
)
from phase2.similarity.store import SemanticRelationshipStore
from phase2.similarity.threshold import ThresholdRecommender


class SimilarityEngine:
    """High-level orchestrator for the similarity pipeline.

    Flow:
        Embeddings → VectorIndex → Pairwise Search → RelationshipBuilder
        → ConfidencePolicy → FilterPipeline → Store → Report
    """

    def __init__(self, config: SimilarityEngineConfig) -> None:
        self._config = config
        self._provider: SimilarityProvider | None = None
        self._store = SemanticRelationshipStore(config.output_directory)
        self._searcher = RelationshipSearcher(self._store)

    @property
    def provider(self) -> SimilarityProvider:
        if self._provider is None:
            self._provider = create_provider(self._config)
        return self._provider

    @property
    def store(self) -> SemanticRelationshipStore:
        return self._store

    def _create_filter_pipeline(self) -> FilterPipeline:
        return FilterPipeline([
            SelfSimilarityFilter(),
            DuplicateRelationshipFilter(self._config.store_bidirectional),
            ThresholdFilter(self._config.similarity_threshold),
            RelationshipPolicyFilter(self._config.allowed_relationships),
            TopKPerSourceFilter(self._config.top_k),
        ])

    def generate(self, force: bool = False) -> dict[str, Any]:
        """Run the full relationship generation pipeline."""
        if not force and self._store.exists():
            count = self._store.count()
            if count > 0:
                return {
                    "total_relationships": count,
                    "status": "skipped",
                    "reason": "existing relationships found (use --force to recompute)",
                }

        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Load all embedding assets
        embeddings_by_type: dict[SourceType, pl.DataFrame] = {}
        source_ids_by_type: dict[SourceType, list[str]] = {}
        for stype, path in self._config.source_paths.items():
            if not path.exists():
                continue
            df = pl.read_parquet(str(path))
            if df.height == 0:
                continue
            embeddings_by_type[stype] = df
            source_ids_by_type[stype] = df["source_id"].to_list()

        if not embeddings_by_type:
            return {
                "total_relationships": 0,
                "status": "error",
                "reason": "no embedding assets found",
            }

        # Build VectorIndex for each source type
        indexes: dict[SourceType, LinearIndex] = {}
        for stype, df in embeddings_by_type.items():
            vecs = np.stack(df["embedding"].to_list()).astype(np.float32)
            indexes[stype] = LinearIndex(vecs)

        # Generate relationships for each source type
        all_relationships: list[SemanticRelationship] = []
        builder = RelationshipBuilder(self._config, self.provider)
        confidence = RelationshipConfidencePolicy(self._config)
        pre_filter_scores: list[float] = []

        for source_type, source_df in embeddings_by_type.items():
            source_vecs = np.stack(source_df["embedding"].to_list()).astype(np.float32)
            source_ids = source_df["source_id"].to_list()
            allowed_targets = self._config.allowed_relationships.get(source_type, [])

            # Build unified index of all allowed target types
            target_dfs: list[tuple[SourceType, pl.DataFrame]] = []
            for ttype in allowed_targets:
                if ttype in embeddings_by_type:
                    target_dfs.append((ttype, embeddings_by_type[ttype]))

            if not target_dfs:
                continue

            # Concatenate target vectors
            all_target_vecs_list = []
            all_target_ids_list = []
            all_target_types_list = []
            for ttype, tdf in target_dfs:
                vecs = np.stack(tdf["embedding"].to_list()).astype(np.float32)
                all_target_vecs_list.append(vecs)
                all_target_ids_list.extend(tdf["source_id"].to_list())
                all_target_types_list.extend([ttype] * tdf.height)

            target_index = LinearIndex(np.vstack(all_target_vecs_list))
            target_ids = all_target_ids_list
            target_types = all_target_types_list

            # Compute frequencies for confidence
            all_ids = source_ids + target_ids
            freq_map = count_frequencies(all_ids)

            # Batch search
            top_k = min(self._config.top_k, target_index.size)
            for i in range(0, len(source_vecs), self._config.batch_size):
                batch_vecs = source_vecs[i : i + self._config.batch_size]
                batch_ids = source_ids[i : i + self._config.batch_size]

                all_scores, all_indices = target_index.search_batch(batch_vecs, k=top_k)

                for q_idx, qid in enumerate(batch_ids):
                    row_scores = all_scores[q_idx]
                    row_indices = all_indices[q_idx]

                    for j in range(len(row_indices)):
                        idx = row_indices[j]
                        sim = max(0.0, min(1.0, float(row_scores[j])))
                        ttype = target_types[idx]
                        tid = target_ids[idx]

                        # Record pre-filter score for threshold analysis
                        pre_filter_scores.append(sim)

                        # Compute confidence before creating relationship
                        conf = confidence.compute(
                            similarity_score=sim,
                            source_frequency=freq_map.get(qid, 1),
                            target_frequency=freq_map.get(tid, 1),
                        )

                        rel = builder.build(
                            source_type=source_type,
                            source_id=qid,
                            target_type=ttype,
                            target_id=tid,
                            similarity_score=sim,
                            confidence=conf,
                        )
                        all_relationships.append(rel)

        # Record pre-filter count
        total_pre_filter = len(all_relationships)

        # Apply filter pipeline (capture stage counts)
        pipeline = self._create_filter_pipeline()
        stage_counts: dict[str, int] = {}
        filtered = all_relationships
        for f in pipeline._filters:
            name = type(f).__name__
            before = len(filtered)
            filtered = f.apply(filtered)
            stage_counts[name] = before - len(filtered)
        stage_counts["total_removed"] = total_pre_filter - len(filtered)
        stage_counts["total_survived"] = len(filtered)

        # Store results
        if filtered:
            self._store.save(filtered)

        elapsed = time.perf_counter() - start

        # Compute basic stats
        stats = compute_stats(
            self._store.load_df(),
            total_source_items=sum(len(ids) for ids in source_ids_by_type.values()),
        )

        # Compute expanded statistics
        df = self._store.load_df() if filtered else pl.DataFrame()
        total_source_items = sum(len(ids) for ids in source_ids_by_type.values())
        expanded_stats = compute_relationship_statistics(
            df, total_source_items=total_source_items
        ) if filtered else RelationshipStatistics()

        # Threshold recommendation
        recommender = ThresholdRecommender()
        threshold_rec = recommender.recommend(
            scores=pre_filter_scores,
            configured_threshold=self._config.similarity_threshold,
        )

        # Generate manifest
        manifest = RelationshipManifest(
            embedding_model=self._config.embedding_model,
            embedding_fingerprint=self._config.model_fingerprint,
            metric=self._config.metric,
            threshold=self._config.similarity_threshold,
            record_count=stats.total_relationships,
            source_counts=stats.source_type_counts,
            target_counts=stats.target_type_counts,
            generated_at=timestamp,
            elapsed_seconds=round(elapsed, 2),
        )
        write_manifest(manifest, self._config.output_directory)

        # Generate text report
        generate_quality_report(
            stats,
            self._config.output_directory,
            elapsed,
            threshold_rec=threshold_rec,
            filter_counts=stage_counts,
            configured_threshold=self._config.similarity_threshold,
        )

        # Generate JSON report
        write_json_report(
            expanded_stats,
            self._config.output_directory,
            elapsed,
            threshold_rec=threshold_rec,
            filter_counts=stage_counts,
            df=df if filtered else None,
        )

        return {
            "total_relationships": stats.total_relationships,
            "unique_source_ids": stats.unique_source_ids,
            "unique_target_ids": stats.unique_target_ids,
            "unique_pairs": stats.unique_pair_ids,
            "avg_similarity": round(stats.avg_similarity, 6),
            "avg_confidence": round(stats.avg_confidence, 6),
            "avg_neighbors": expanded_stats.average_neighbors,
            "density": round(stats.density, 8),
            "threshold_warning": threshold_rec.threshold_warning,
            "elapsed_seconds": round(elapsed, 2),
        }

    def stats(self) -> RelationshipStats:
        """Return aggregate stats over stored relationships."""
        df = self._store.load_df()
        return compute_stats(df)

    def detailed_stats(self) -> RelationshipStatistics:
        """Return expanded statistics over stored relationships."""
        df = self._store.load_df()
        return compute_relationship_statistics(df)

    def search_relationships(self, source_id: str, k: int = 10) -> list[SemanticRelationship]:
        """Search relationships by source_id."""
        return self._searcher.search_by_source(source_id, k=k)

    def verify(self) -> dict[str, Any]:
        """Verify integrity of stored relationships."""
        df = self._store.load_df()
        result: dict[str, Any] = {"valid": True, "checks": {}}

        if df.height == 0:
            result["valid"] = False
            result["checks"]["has_relationships"] = "FAIL: no relationships found"
            return result

        result["checks"]["has_relationships"] = f"PASS ({df.height} relationships)"

        ids = df["relationship_id"].to_list()
        unique = len(set(ids)) == len(ids)
        result["checks"]["unique_ids"] = "PASS" if unique else "FAIL"
        if not unique:
            result["valid"] = False

        sims = df["similarity_score"].to_numpy()
        in_range = float(sims.min()) >= 0.0 and float(sims.max()) <= 1.0
        result["checks"]["similarity_range"] = "PASS" if in_range else "WARN"

        confs = df["confidence"].to_numpy()
        conf_range = float(confs.min()) >= 0.0 and float(confs.max()) <= 1.0
        result["checks"]["confidence_range"] = "PASS" if conf_range else "WARN"

        result["total_relationships"] = df.height
        return result