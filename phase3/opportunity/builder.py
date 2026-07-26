"""OpportunityBuilder — end-to-end opportunity discovery pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from phase3.opportunity.cache import OpportunityCache
from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.extractor import OpportunityExtractor
from phase3.opportunity.providers.registry import (
    available_ranking_providers,
    available_scoring_providers,
)
from phase3.opportunity.ranking import OpportunityRanker
from phase3.opportunity.recommendation import RecommendationEngine
from phase3.opportunity.scoring import OpportunityScorer
from phase3.opportunity.store import OpportunityStore
from phase3.opportunity.validator import OpportunityValidator


class OpportunityBuilder:
    """End-to-end pipeline: extract -> score -> rank -> recommend -> validate -> persist."""

    def __init__(self, config: OpportunityConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or _generate_run_id()
        self._store = OpportunityStore(config.output_dir)
        self._cache = OpportunityCache(self._store, config)
        self._extractor = OpportunityExtractor(config)
        self._scorer = OpportunityScorer(config)
        self._ranker = OpportunityRanker(config)
        self._recommender = RecommendationEngine(config)
        self._validator = OpportunityValidator()

    @property
    def store(self) -> OpportunityStore:
        return self._store

    @property
    def config(self) -> OpportunityConfig:
        return self._config

    def build(
        self,
        root_causes: list | None = None,
        evidence_aggregations: list | None = None,
        inferences: list | None = None,
        chains: list | None = None,
        kg_nodes: list | None = None,
        kg_edges: list | None = None,
        clusters: list | None = None,
        reasoning_checksums: dict | None = None,
        kg_checksums: dict | None = None,
        force: bool = False,
    ) -> dict:
        start = time.perf_counter()

        # Guard: no data
        if not root_causes and not evidence_aggregations and not clusters:
            return self._empty_result(start)

        # Check cache
        if not force and self._cache.is_valid(reasoning_checksums, kg_checksums):
            cached = self._cache.load()
            if cached and cached.opportunities:
                meta = cached.metadata
                if meta:
                    meta = meta.model_copy(update={"cache_hit": True})
                    self._store.save_metadata(meta)
                return self._build_result(cached.opportunities, meta, start, from_cache=True)

        # Stage 1: Extract candidates
        candidates = self._extractor.extract(
            root_causes=root_causes or [],
            evidence_aggregations=evidence_aggregations or [],
            inferences=inferences or [],
            chains=chains or [],
            kg_nodes=kg_nodes or [],
            kg_edges=kg_edges or [],
            clusters=clusters or [],
        )

        # Build scoring context
        context = self._build_context(
            candidates=candidates,
            root_causes=root_causes or [],
            evidence_aggregations=evidence_aggregations or [],
            inferences=inferences or [],
            chains=chains or [],
            clusters=clusters or [],
            kg_nodes=kg_nodes or [],
            kg_edges=kg_edges or [],
        )

        # Stage 2: Score
        opportunities = self._scorer.score(candidates, context, self._run_id)

        # Stage 3: Rank
        opportunities = self._ranker.rank(opportunities, context)

        # Stage 4: Recommend
        opportunities = self._recommender.recommend(opportunities, context)

        # Stage 5: Validate
        chain_ids = {c.chain_id for c in (chains or []) if hasattr(c, "chain_id")}
        validation = self._validator.validate(opportunities, valid_chain_ids=chain_ids)

        # Build metadata
        meta = self._build_metadata(opportunities, candidates, start)
        self._store.save_opportunities(opportunities, self._run_id)
        self._store.save_metadata(meta)

        # Cache
        self._cache.save(reasoning_checksums, kg_checksums)

        # Manifest
        manifest = self._build_manifest(opportunities, candidates, context, validation, start)
        self._store.save_manifest(manifest)

        return self._build_result(opportunities, meta, start, from_cache=False)

    def _build_context(self, **kwargs) -> dict[str, Any]:
        candidates = kwargs.get("candidates", [])
        root_causes = kwargs.get("root_causes", [])
        evidence = kwargs.get("evidence_aggregations", [])
        clusters = kwargs.get("clusters", [])
        kg_nodes = kwargs.get("kg_nodes", [])
        kg_edges = kwargs.get("kg_edges", [])

        all_products: set[str] = set()
        all_companies: set[str] = set()
        all_technologies: set[str] = set()
        for n in kg_nodes:
            nt = n.node_type.value if hasattr(n, "node_type") and hasattr(n.node_type, "value") else ""
            label = n.label if hasattr(n, "label") else ""
            if nt == "product":
                all_products.add(label)
            elif nt == "company":
                all_companies.add(label)
            elif nt == "technology":
                all_technologies.add(label)

        max_evidence = max((c.get("evidence_count", 0) for c in candidates), default=1)
        max_products = max((c.get("product_count", 0) for c in candidates), default=1)
        total_platforms = max(
            (c.get("platform_count", 0) for c in candidates), default=1
        )

        return {
            "max_evidence_count": max_evidence,
            "max_product_count": max_products,
            "total_platforms": total_platforms,
            "total_products": len(all_products),
            "total_companies": len(all_companies),
            "total_technologies": len(all_technologies),
            "root_cause_count": len(root_causes),
            "evidence_count": len(evidence),
            "cluster_count": len(clusters),
            "kg_node_count": len(kg_nodes),
            "kg_edge_count": len(kg_edges),
        }

    def _build_metadata(self, opportunities: list, candidates: list, start: float) -> Any:
        from phase3.opportunity.schema import OpportunityMetadata
        scores = [o.opportunity_score for o in opportunities] if opportunities else [0.0]
        rec_dist: dict[str, int] = {}
        bm_dist: dict[str, int] = {}
        for o in opportunities:
            rec_dist[o.recommendation_type.value] = rec_dist.get(o.recommendation_type.value, 0) + 1
            bm_dist[o.suggested_business_model.value] = bm_dist.get(o.suggested_business_model.value, 0) + 1

        return OpportunityMetadata(
            run_id=self._run_id,
            total_candidates=len(candidates),
            total_opportunities=len(opportunities),
            scored_opportunities=sum(1 for o in opportunities if o.status.value in ("scored", "ranked", "recommended")),
            ranked_opportunities=sum(1 for o in opportunities if o.status.value in ("ranked", "recommended")),
            scoring_providers_used=self._scorer.providers_used,
            business_model_providers_used=self._recommender.providers_used,
            recommendation_distribution=rec_dist,
            business_model_distribution=bm_dist,
            avg_opportunity_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            min_opportunity_score=min(scores) if scores else 0.0,
            max_opportunity_score=max(scores) if scores else 0.0,
            elapsed_seconds=round(time.perf_counter() - start, 4),
            pipeline_version=self._config.version,
            schema_version="1.0",
        )

    def _build_manifest(
        self, opportunities: list, candidates: list, context: dict, validation: Any, start: float
    ) -> dict:
        return {
            "run_id": self._run_id,
            "pipeline_version": self._config.version,
            "opportunity_version": self._config.opportunity_version,
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "total_candidates": len(candidates),
            "total_opportunities": len(opportunities),
            "scoring_providers": self._scorer.providers_used,
            "business_model_providers": self._recommender.providers_used,
            "config": self._config.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json") if hasattr(validation, "model_dump") else {},
            "checksums": self._store.checksums(),
        }

    def _build_result(self, opportunities: list, metadata: Any, start: float, from_cache: bool = False) -> dict:
        return {
            "run_id": self._run_id,
            "total_candidates": metadata.total_candidates if metadata else 0,
            "total_opportunities": len(opportunities),
            "scored_opportunities": metadata.scored_opportunities if metadata else 0,
            "ranked_opportunities": metadata.ranked_opportunities if metadata else 0,
            "avg_opportunity_score": metadata.avg_opportunity_score if metadata else 0.0,
            "top_score": metadata.max_opportunity_score if metadata else 0.0,
            "recommendation_distribution": metadata.recommendation_distribution if metadata else {},
            "business_model_distribution": metadata.business_model_distribution if metadata else {},
            "scoring_providers": metadata.scoring_providers_used if metadata else [],
            "business_model_providers": metadata.business_model_providers_used if metadata else [],
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "cache_hit": from_cache or (metadata.cache_hit if metadata else False),
        }

    def _empty_result(self, start: float) -> dict:
        return {
            "run_id": self._run_id,
            "total_candidates": 0,
            "total_opportunities": 0,
            "scored_opportunities": 0,
            "ranked_opportunities": 0,
            "avg_opportunity_score": 0.0,
            "top_score": 0.0,
            "recommendation_distribution": {},
            "business_model_distribution": {},
            "scoring_providers": [],
            "business_model_providers": [],
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "cache_hit": False,
        }


def _generate_run_id() -> str:
    import hashlib
    import time
    raw = f"opp-{time.time_ns()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
