"""SemanticEdgeBuilder — builds SIMILAR_TO edges from semantic_relationships.parquet."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from phase2.knowledge_graph.edge_builders.base import EdgeBuilder
from phase2.knowledge_graph.registry import register_edge_builder
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode


@register_edge_builder("semantic")
class SemanticEdgeBuilder(EdgeBuilder):
    """Builds SIMILAR_TO edges from semantic_relationships.parquet.

    Creates an edge between two nodes if both exist in the graph
    and the similarity score meets the minimum threshold.
    """

    def __init__(self, asset_path: Path, config: Any = None) -> None:
        self._asset_path = Path(asset_path)
        self._config = config
        self._source_asset = "semantic_relationships.parquet"
        self._min_weight = getattr(config, "minimum_weight", 0.1) if config else 0.1

    def build_edges(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        if not self._asset_path.exists():
            return []
        df = pl.read_parquet(str(self._asset_path))
        if df.height == 0:
            return []

        node_map: dict[str, str] = {}
        for node in nodes:
            node_map[node.source_id] = node.node_id
            node_map[node.node_id] = node.node_id

        edges: list[GraphEdge] = []
        now = datetime.now(timezone.utc).isoformat()

        for row in df.to_dicts():
            source_source_id = str(row.get("source_id", ""))
            target_source_id = str(row.get("target_id", ""))
            source_nid = node_map.get(source_source_id)
            target_nid = node_map.get(target_source_id)
            if source_nid is None or target_nid is None:
                continue
            weight = float(row.get("similarity_score", row.get("weight", 0.5)))
            if weight < self._min_weight:
                continue

            raw = f"{EdgeType.SIMILAR_TO.value}:{source_nid}:{target_nid}:{PIPELINE_VERSION}:{SCHEMA_VERSION}:1.0"
            edge_id = hashlib.sha256(raw.encode()).hexdigest()

            confidence = float(row.get("confidence", weight))

            edge = GraphEdge(
                edge_id=edge_id,
                source_node_id=source_nid,
                target_node_id=target_nid,
                edge_type=EdgeType.SIMILAR_TO,
                weight=weight,
                confidence=confidence,
                properties={"similarity_score": weight},
                metadata={"source_asset": self._source_asset},
                attributes={},
                source_asset=self._source_asset,
                created_at=now,
                pipeline_version=PIPELINE_VERSION,
                schema_version=SCHEMA_VERSION,
            )
            edges.append(edge)

        return edges
