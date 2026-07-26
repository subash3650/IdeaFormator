"""OpportunityEngine — facade for the Opportunity Discovery Engine."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.builder import OpportunityBuilder
from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.providers.registry import available_scoring_providers
from phase3.opportunity.store import OpportunityStore


class OpportunityEngine:
    """High-level facade for the Opportunity Discovery Engine.

    Lazily loads upstream data and orchestrates the builder.
    """

    def __init__(self, config: OpportunityConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id
        self._builder = OpportunityBuilder(config, run_id=run_id)

    @property
    def store(self) -> OpportunityStore:
        return self._builder.store

    @property
    def config(self) -> OpportunityConfig:
        return self._config

    def discover(self, force: bool = False) -> dict:
        """Run full opportunity discovery pipeline.

        Loads reasoning and KG outputs, then delegates to builder.
        """
        from phase2.reasoning.store import ReasoningStore
        from phase2.knowledge_graph.store import KnowledgeGraphStore

        knowledge_dir = self._config.knowledge_dir or self._config.output_dir

        # Load reasoning outputs
        rstore = ReasoningStore(knowledge_dir)
        root_causes = rstore.load_root_causes()
        evidence = rstore.load_evidence_aggregations()
        inferences = rstore.load_inferences()
        chains = rstore.load_chains()
        reasoning_checksums = rstore.checksums()

        # Load knowledge graph
        kg_store = KnowledgeGraphStore(knowledge_dir)
        kg_nodes = kg_store.load_nodes()
        kg_edges = kg_store.load_edges()
        kg_checksums = {"nodes": "", "edges": ""}
        try:
            cs = kg_store.checksums()
            kg_checksums = {"nodes": cs.get("nodes", ""), "edges": cs.get("edges", "")}
        except Exception:
            pass

        # Load clusters
        clusters = self._load_clusters(knowledge_dir)

        return self._builder.build(
            root_causes=root_causes,
            evidence_aggregations=evidence,
            inferences=inferences,
            chains=chains,
            kg_nodes=kg_nodes,
            kg_edges=kg_edges,
            clusters=clusters,
            reasoning_checksums=reasoning_checksums,
            kg_checksums=kg_checksums,
            force=force,
        )

    def stats(self) -> dict:
        """Return statistics about stored opportunities."""
        opportunities = self._builder.store.load_opportunities()
        metadata = self._builder.store.load_metadata()
        scores = [o.opportunity_score for o in opportunities] if opportunities else [0.0]
        rec_dist: dict[str, int] = {}
        bm_dist: dict[str, int] = {}
        for o in opportunities:
            rec_dist[o.recommendation_type.value] = rec_dist.get(o.recommendation_type.value, 0) + 1
            bm_dist[o.suggested_business_model.value] = bm_dist.get(o.suggested_business_model.value, 0) + 1

        return {
            "total_opportunities": len(opportunities),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "recommendation_distribution": rec_dist,
            "business_model_distribution": bm_dist,
            "run_id": metadata.run_id if metadata else "",
            "cache_hit": metadata.cache_hit if metadata else False,
            "elapsed_seconds": metadata.elapsed_seconds if metadata else 0.0,
        }

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search stored opportunities by text."""
        from phase3.opportunity.search import OpportunitySearch

        opportunities = self._builder.store.load_opportunities()
        searcher = OpportunitySearch(opportunities)
        results = searcher.search_text(query, top_k=top_k)
        return [o.model_dump(mode="json") for o in results]

    def clear_cache(self) -> None:
        """Invalidate opportunity cache."""
        self._builder._cache.invalidate()

    @staticmethod
    def _load_clusters(knowledge_dir: Path) -> list:
        """Load semantic clusters from the cluster store."""
        try:
            from phase2.clustering.store import ClusterStore
            from phase2.clustering.config import ClusteringConfig

            cfg = ClusteringConfig(output_directory=knowledge_dir)
            cstore = ClusterStore(cfg)
            return cstore.load()
        except Exception:
            return []
