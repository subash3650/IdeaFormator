"""ClusterNodeBuilder — builds CLUSTER nodes from semantic_clusters.parquet."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from phase2.knowledge_graph.node_builders.base import NodeBuilder
from phase2.knowledge_graph.registry import register_node_builder
from phase2.knowledge_graph.schema import GraphNode, NodeType


@register_node_builder("cluster")
class ClusterNodeBuilder(NodeBuilder):
    """Builds CLUSTER nodes from semantic_clusters.parquet."""

    def __init__(self, asset_path: Path, config: Any = None) -> None:
        self._asset_path = Path(asset_path)
        self._config = config
        self._source_asset = "semantic_clusters.parquet"

    def build_nodes(self) -> list[GraphNode]:
        if not self._asset_path.exists():
            return []
        df = pl.read_parquet(str(self._asset_path))
        if df.height == 0:
            return []

        nodes: list[GraphNode] = []
        now = datetime.now(timezone.utc).isoformat()

        for row in df.to_dicts():
            source_id = str(row.get("cluster_id", row.get("id", "")))
            raw = f"{NodeType.CLUSTER.value}:{self._source_asset}:{source_id}:{PIPELINE_VERSION}:{SCHEMA_VERSION}"
            node_id = hashlib.sha256(raw.encode()).hexdigest()
            label = f"Cluster: {row.get('representative_id', source_id)[:50]}"[:200]

            properties = {
                k: v for k, v in row.items()
                if k not in ("cluster_id", "id", "representative_id", "member_ids", "confidence", "metadata")
            }
            properties["representative_id"] = str(row.get("representative_id", ""))
            members = row.get("member_ids", [])
            if members:
                properties["member_ids"] = list(members) if hasattr(members, "__iter__") and not isinstance(members, str) else [str(members)]

            meta = {
                "source_asset": self._source_asset,
                "source_id": source_id,
                "run_id": "",
            }

            confidence = float(row.get("quality_score", row.get("confidence", 1.0)))

            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.CLUSTER,
                label=label,
                properties=properties,
                metadata=meta,
                attributes={},
                source_asset=self._source_asset,
                source_id=source_id,
                confidence=confidence,
                created_at=now,
                pipeline_version=PIPELINE_VERSION,
                schema_version=SCHEMA_VERSION,
            )
            nodes.append(node)

        return nodes
