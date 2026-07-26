"""HierarchicalEdgeBuilder — builds BELONGS_TO and MEMBER_OF_CLUSTER edges."""

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


@register_edge_builder("hierarchical")
class HierarchicalEdgeBuilder(EdgeBuilder):
    """Builds MEMBER_OF_CLUSTER edges from semantic_clusters.parquet.

    Creates an edge from each member node to its containing cluster node.
    Also creates BELONGS_TO edges for the representative-to-cluster relationship.
    """

    def __init__(self, asset_path: Path, config: Any = None) -> None:
        self._asset_path = Path(asset_path)
        self._config = config
        self._member_source = "semantic_clusters.parquet"
        self._min_confidence = getattr(config, "minimum_confidence", 0.5) if config else 0.5

    def build_edges(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        if not self._asset_path.exists():
            return []
        df = pl.read_parquet(str(self._asset_path))
        if df.height == 0:
            return []

        # Build a map from source_id -> node_id for membership resolution
        node_by_source_id: dict[str, str] = {}
        for node in nodes:
            node_by_source_id[node.source_id] = node.node_id

        # Build a map from cluster_id (source_id) -> cluster node_id
        cluster_nodes: dict[str, str] = {}
        for node in nodes:
            if node.node_type.value == "cluster":
                cluster_nodes[node.source_id] = node.node_id

        edges: list[GraphEdge] = []
        now = datetime.now(timezone.utc).isoformat()

        for row in df.to_dicts():
            cluster_source_id = str(row.get("cluster_id", ""))
            cluster_nid = cluster_nodes.get(cluster_source_id)
            if cluster_nid is None:
                continue

            representative_id = str(row.get("representative_id", ""))
            member_ids_raw = row.get("member_ids", [])
            if hasattr(member_ids_raw, "__iter__") and not isinstance(member_ids_raw, str):
                all_member_ids: list[str] = list(member_ids_raw)
            else:
                all_member_ids = [str(member_ids_raw)] if member_ids_raw else []

            for member_source_id in all_member_ids:
                member_nid = node_by_source_id.get(member_source_id)
                if member_nid is None:
                    continue

                edge_type = EdgeType.MEMBER_OF_CLUSTER
                if member_source_id == representative_id:
                    edge_type = EdgeType.BELONGS_TO

                raw = f"{edge_type.value}:{member_nid}:{cluster_nid}:{PIPELINE_VERSION}:{SCHEMA_VERSION}:1.0"
                edge_id = hashlib.sha256(raw.encode()).hexdigest()

                confidence = 1.0 if member_source_id == representative_id else 0.8

                edge = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=member_nid,
                    target_node_id=cluster_nid,
                    edge_type=edge_type,
                    weight=1.0,
                    confidence=confidence,
                    properties={},
                    metadata={
                        "source_asset": self._member_source,
                        "cluster_id": cluster_source_id,
                        "representative": str(member_source_id == representative_id).lower(),
                    },
                    attributes={},
                    source_asset=self._member_source,
                    created_at=now,
                    pipeline_version=PIPELINE_VERSION,
                    schema_version=SCHEMA_VERSION,
                )
                edges.append(edge)

        return edges
