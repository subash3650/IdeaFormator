"""ReasoningEngine facade — orchestrates the full reasoning pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.store import KnowledgeGraphStore
from phase2.reasoning.cache import ReasoningCache
from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.inference import InferenceEngine
from phase2.reasoning.provenance_id import generate_run_id
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.schema import InferenceOutput
from phase2.reasoning.store import ReasoningStore


class ReasoningEngine:
    def __init__(self, config: ReasoningConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or generate_run_id()
        self._store = ReasoningStore(config.output_dir)
        self._kg_store = KnowledgeGraphStore(config.output_dir)
        self._cache = ReasoningCache(self._store, config)
        self._rule_engine = RuleEngine(
            enabled_rules=list(config.enabled_rules),
            max_iterations=config.max_rule_iterations,
            max_inferences=config.max_inferences_per_run,
        )
        self._rule_engine.initialize()
        self._propagator = ConfidencePropagator(
            strategy=config.propagation_strategy,
            decay_rate=config.decay_rate,
            min_confidence=config.min_confidence,
        )
        self._chain_tracker = ChainTracker(run_id=self._run_id)
        self._evidence_aggregator = EvidenceAggregator(
            min_evidence_count=config.min_evidence_count,
            conflicting_threshold=config.conflicting_evidence_threshold,
        )
        self._root_cause_discoverer = RootCauseDiscoverer(
            ranking=config.root_cause_ranking,
            max_depth=config.max_root_cause_depth,
            min_confidence=config.min_confidence,
        )
        self._explanation_generator = ExplanationGenerator()
        self._inference_engine = InferenceEngine(
            config=config,
            rule_engine=self._rule_engine,
            propagator=self._propagator,
            chain_tracker=self._chain_tracker,
            evidence_aggregator=self._evidence_aggregator,
            root_cause_discoverer=self._root_cause_discoverer,
            explanation_generator=self._explanation_generator,
        )
        self._graph: GraphInterface | None = None

    @property
    def store(self) -> ReasoningStore:
        return self._store

    @property
    def config(self) -> ReasoningConfig:
        return self._config

    def reason(self, force: bool = False) -> dict:
        start = time.perf_counter()
        graph = self._load_graph()

        cache_valid = self._cache.is_valid(graph)
        if cache_valid and not force:
            cached = self._cache.load()
            if cached and cached.metadata:
                cached.metadata = cached.metadata.model_copy(update={"cache_hit": True})
                self._store.save_metadata(cached.metadata)
                return self._build_result(cached, from_cache=True)

        output = self._inference_engine.infer(graph, self._run_id)
        self._cache.save(graph, output)
        if output.metadata:
            output = output.model_copy(update={
                "metadata": output.metadata.model_copy(update={"cache_hit": False}),
            })

        manifest = self._build_manifest(graph, output, start)
        self._store.save_manifest(manifest)
        elapsed = time.perf_counter() - start

        return self._build_result(output)

    def _build_result(self, output: InferenceOutput, from_cache: bool = False) -> dict:
        meta = output.metadata
        return {
            "run_id": self._run_id,
            "inference_count": len(output.inferences),
            "chain_count": len(output.chains),
            "root_cause_count": len(output.root_causes),
            "evidence_aggregation_count": len(output.evidence_aggregations),
            "explanation_count": len(output.explanations),
            "rules_applied": meta.rules_applied if meta else [],
            "rule_firing_counts": meta.rule_firing_counts if meta else {},
            "elapsed_seconds": meta.elapsed_seconds if meta else 0.0,
            "cache_hit": from_cache or (meta.cache_hit if meta else False),
        }

    def _build_manifest(
        self, graph: GraphInterface, output: InferenceOutput, start: float
    ) -> dict:
        node_count = graph.node_count()
        edge_count = graph.edge_count()
        kg_run_id = ""
        try:
            kg_run_id = self._kg_store.run_id()
        except Exception:
            kg_run_id = ""
        elapsed = round(time.perf_counter() - start, 4)
        return {
            "run_id": self._run_id,
            "kg_run_id": kg_run_id,
            "kg_node_count": node_count,
            "kg_edge_count": edge_count,
            "pipeline_version": self._config.version,
            "reasoning_version": self._config.reasoning_version,
            "generated_at": output.metadata.created_at if output.metadata else "",
            "elapsed_seconds": elapsed,
            "inference_count": len(output.inferences),
            "chain_count": len(output.chains),
            "root_cause_count": len(output.root_causes),
            "config": self._config.model_dump(mode="json"),
            "rules_applied": output.metadata.rules_applied if output.metadata else [],
            "rule_firing_counts": output.metadata.rule_firing_counts if output.metadata else {},
        }

    def _load_graph(self) -> GraphInterface:
        if self._graph is not None:
            return self._graph
        graph = CustomGraph()
        nodes_path = self._kg_store.nodes_path
        edges_path = self._kg_store.edges_path
        if nodes_path.exists():
            df = __import__("polars").read_parquet(nodes_path)
            for row in df.iter_rows(named=True):
                try:
                    from phase2.knowledge_graph.schema import GraphNode, NodeType
                    node = GraphNode(
                        node_id=row["node_id"],
                        node_type=NodeType(row.get("node_type", "observation")),
                        label=row.get("label", ""),
                        properties=json.loads(row.get("properties") or "{}"),
                        metadata=json.loads(row.get("metadata") or "{}"),
                        attributes=json.loads(row.get("attributes") or "{}"),
                        source_asset=row.get("source_asset", ""),
                        source_id=row.get("source_id", ""),
                        confidence=float(row.get("confidence", 1.0)),
                        pipeline_version=row.get("pipeline_version", "1.0"),
                        schema_version=row.get("schema_version", "1.0"),
                    )
                    graph.add_node(node)
                except Exception:
                    continue
        if edges_path.exists():
            import json
            df = __import__("polars").read_parquet(edges_path)
            for row in df.iter_rows(named=True):
                try:
                    from phase2.knowledge_graph.schema import EdgeType, GraphEdge
                    edge = GraphEdge(
                        edge_id=row["edge_id"],
                        source_node_id=row["source_node_id"],
                        target_node_id=row["target_node_id"],
                        edge_type=EdgeType(row.get("edge_type", "similar_to")),
                        weight=float(row.get("weight", 0.5)),
                        confidence=float(row.get("confidence", 0.5)),
                        properties=json.loads(row.get("properties") or "{}"),
                        metadata=json.loads(row.get("metadata") or "{}"),
                        attributes=json.loads(row.get("attributes") or "{}"),
                        source_asset=row.get("source_asset", ""),
                        pipeline_version=row.get("pipeline_version", "1.0"),
                        schema_version=row.get("schema_version", "1.0"),
                    )
                    graph.add_edge(edge)
                except Exception:
                    continue
        self._graph = graph
        return graph

    def clear_cache(self) -> None:
        self._cache.invalidate()

    def stats(self) -> dict:
        graph = self._load_graph()
        cache_valid = self._cache.is_valid(graph)
        inferences = self._store.load_inferences()
        chains = self._store.load_chains()
        root_causes = self._store.load_root_causes()
        return {
            "graph_nodes": graph.node_count(),
            "graph_edges": graph.edge_count(),
            "inferences": len(inferences),
            "chains": len(chains),
            "root_causes": len(root_causes),
            "cache_valid": cache_valid,
            "run_id": self._run_id,
        }
