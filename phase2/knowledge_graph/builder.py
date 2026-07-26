"""KnowledgeGraphBuilder — orchestrates node and edge construction from pipeline assets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pain_intelligence.knowledge.manifest import PipelineManifest, compute_checksum, generate_run_id
from pain_intelligence.knowledge.metadata import read_parquet_metadata
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.edge_builders.base import EdgeBuilder
from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.node_builders.base import NodeBuilder
from phase2.knowledge_graph.registry import create_edge_builder, create_node_builder
from phase2.knowledge_graph.schema import GraphEdge, GraphNode
from phase2.knowledge_graph.store import KnowledgeGraphStore
from phase2.knowledge_graph.validator import GraphValidator


_REQUIRED_ASSETS: dict[str, str] = {
    "observation": "observations.parquet",
    "evidence": "evidence.parquet",
    "problem_signal": "problem_signals.parquet",
    "cluster": "semantic_clusters.parquet",
    "semantic": "semantic_relationships.parquet",
    "hierarchical": "semantic_clusters.parquet",
    "causal": "problem_signals.parquet",
}


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from existing pipeline assets.

    Flow:
        1. Validate upstream assets exist
        2. Run registered NodeBuilders
        3. Deduplicate nodes
        4. Run registered EdgeBuilders
        5. Validate graph integrity
        6. Save to KnowledgeGraphStore
        7. Return in-memory CustomGraph
    """

    def __init__(self, config: KnowledgeGraphConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or generate_run_id()
        self._store = KnowledgeGraphStore(config.output_dir)
        self._validator = GraphValidator()
        self._manifest = PipelineManifest(
            config.output_dir.parent.parent if config.output_dir.parent.parent.exists() else Path("pain_intelligence/knowledge"),
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def store(self) -> KnowledgeGraphStore:
        return self._store

    def build(self, force: bool = False) -> CustomGraph:
        start = time.perf_counter()

        node_builders: list[NodeBuilder] = []
        for node_type in self._config.node_types:
            builder_name = node_type.value
            asset_name = _REQUIRED_ASSETS.get(builder_name)
            if asset_name is None:
                continue
            asset_path = self._config.output_dir / asset_name
            if not force and not asset_path.exists():
                continue
            try:
                nb = create_node_builder(builder_name, asset_path=asset_path, config=self._config)
                node_builders.append(nb)
            except KeyError:
                continue

        edge_builders: list[EdgeBuilder] = []
        for edge_type in self._config.edge_types:
            builder_name = edge_type.value
            asset_name = _REQUIRED_ASSETS.get(builder_name)
            if asset_name is None:
                continue
            asset_path = self._config.output_dir / asset_name
            if not force and not asset_path.exists():
                continue
            try:
                if builder_name == "causal":
                    eb = create_edge_builder(builder_name, signal_asset_path=asset_path, config=self._config)
                else:
                    eb = create_edge_builder(builder_name, asset_path=asset_path, config=self._config)
                edge_builders.append(eb)
            except KeyError:
                continue

        # Build nodes
        all_nodes: list[GraphNode] = []
        for nb in node_builders:
            try:
                all_nodes.extend(nb.build_nodes())
            except Exception:
                continue

        # Deduplicate by node_id (last-write-wins)
        node_map: dict[str, GraphNode] = {}
        for node in all_nodes:
            node_map[node.node_id] = node
        deduped_nodes = list(node_map.values())

        # Build edges
        all_edges: list[GraphEdge] = []
        for eb in edge_builders:
            try:
                all_edges.extend(eb.build_edges(deduped_nodes))
            except Exception:
                continue

        # Filter edges: drop where endpoint missing
        node_ids = set(node_map.keys())
        valid_edges = [e for e in all_edges if e.source_node_id in node_ids and e.target_node_id in node_ids]

        # Apply minimum confidence and weight thresholds
        min_conf = self._config.minimum_confidence
        min_weight = self._config.minimum_weight
        filtered_edges = [
            e for e in valid_edges
            if e.confidence >= min_conf and e.weight >= min_weight
        ]

        # Build graph
        graph = CustomGraph()
        for node in deduped_nodes:
            graph.add_node(node)
        for edge in filtered_edges:
            graph.add_edge(edge)

        # Validate
        validation = self._validator.validate(graph)
        if not validation.valid:
            raise ValueError(f"Graph validation failed: {'; '.join(validation.errors[:5])}")

        # Compute metadata
        meta = graph.metadata(self._run_id)
        meta = meta.model_copy(update={
            "pipeline_version": self._config.version,
            "schema_version": self._config.version,
        })

        # Save
        input_checksum = self._compute_input_checksum()
        self._store.save_nodes(deduped_nodes, self._run_id, input_checksum)
        self._store.save_edges(filtered_edges, self._run_id, input_checksum)
        self._store.save_metadata(meta)

        # Save manifest
        manifest_data: dict[str, Any] = {
            "run_id": self._run_id,
            "pipeline_version": self._config.version,
            "schema_version": self._config.version,
            "builder_version": self._config.builder_version,
            "generated_at": meta.created_at,
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "node_count": len(deduped_nodes),
            "edge_count": len(filtered_edges),
            "input_checksum": input_checksum,
            "config": self._config.model_dump(mode="json"),
            "node_builders": [type(nb).__name__ for nb in node_builders],
            "edge_builders": [type(eb).__name__ for eb in edge_builders],
            "validation": validation.model_dump(mode="json"),
        }
        self._store.save_manifest(manifest_data)

        return graph

    def _compute_input_checksum(self) -> str:
        combined = hashlib_sha256 = __import__("hashlib").sha256()
        for asset_name in _REQUIRED_ASSETS.values():
            path = self._config.output_dir / asset_name
            if path.exists():
                combined.update(path.read_bytes())
        return combined.hexdigest()[:16]
