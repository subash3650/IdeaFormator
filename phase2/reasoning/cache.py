"""Reasoning cache — hash-based caching of reasoning results."""

from __future__ import annotations

import hashlib
from pathlib import Path

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.schema import InferenceOutput
from phase2.reasoning.store import ReasoningStore


class ReasoningCache:
    def __init__(self, store: ReasoningStore, config: ReasoningConfig) -> None:
        self._store = store
        self._config = config
        self._cache_dir = Path(store.base_path) / "reasoning_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def hash_graph(self, graph: GraphInterface) -> str:
        h = hashlib.sha256()
        node_ids = sorted(n.node_id for n in graph.nodes())
        for nid in node_ids:
            h.update(nid.encode())
        edge_ids = sorted(e.edge_id for e in graph.edges())
        for eid in edge_ids:
            h.update(eid.encode())
        h.update(self._config.version.encode())
        h.update(self._config.reasoning_version.encode())
        rule_string = "|".join(sorted(self._config.enabled_rules))
        h.update(rule_string.encode())
        return h.hexdigest()[:16]

    def is_valid(self, graph: GraphInterface) -> bool:
        if not self._config.cache_enabled:
            return False
        marker = self._cache_dir / "cache_marker"
        if not marker.exists():
            return False
        stored_hash = marker.read_text(encoding="utf-8").strip()
        current_hash = self.hash_graph(graph)
        return stored_hash == current_hash

    def save(self, graph: GraphInterface, output: InferenceOutput) -> None:
        current_hash = self.hash_graph(graph)
        run_id = output.metadata.run_id if output.metadata else "cache"
        if output.inferences:
            self._store.save_inferences(output.inferences, run_id)
        if output.chains:
            self._store.save_chains(output.chains, run_id)
        if output.root_causes:
            self._store.save_root_causes(output.root_causes, run_id)
        if output.evidence_aggregations:
            self._store.save_evidence_aggregations(output.evidence_aggregations, run_id)
        if output.explanations:
            self._store.save_explanations(output.explanations)
        if output.metadata:
            self._store.save_metadata(output.metadata)
        marker = self._cache_dir / "cache_marker"
        marker.write_text(current_hash, encoding="utf-8")

    def load(self) -> InferenceOutput | None:
        if not self._config.cache_enabled:
            return None
        marker = self._cache_dir / "cache_marker"
        if not marker.exists():
            return None
        inferences = self._store.load_inferences()
        chains = self._store.load_chains()
        root_causes = self._store.load_root_causes()
        evidence = self._store.load_evidence_aggregations()
        metadata = self._store.load_metadata()
        return InferenceOutput(
            inferences=inferences,
            chains=chains,
            root_causes=root_causes,
            evidence_aggregations=evidence,
            metadata=metadata,
        )

    def invalidate(self) -> None:
        marker = self._cache_dir / "cache_marker"
        if marker.exists():
            marker.unlink()
