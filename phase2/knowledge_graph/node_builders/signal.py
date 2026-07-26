"""SignalNodeBuilder — builds PROBLEM_SIGNAL nodes from problem_signals.parquet."""

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


@register_node_builder("problem_signal")
class SignalNodeBuilder(NodeBuilder):
    """Builds PROBLEM_SIGNAL nodes from problem_signals.parquet."""

    def __init__(self, asset_path: Path, config: Any = None) -> None:
        self._asset_path = Path(asset_path)
        self._config = config
        self._source_asset = "problem_signals.parquet"

    def build_nodes(self) -> list[GraphNode]:
        if not self._asset_path.exists():
            return []
        df = pl.read_parquet(str(self._asset_path))
        if df.height == 0:
            return []

        nodes: list[GraphNode] = []
        now = datetime.now(timezone.utc).isoformat()

        for row in df.to_dicts():
            source_id = str(row.get("signal_id", row.get("problem_signal_id", row.get("id", ""))))
            raw = f"{NodeType.PROBLEM_SIGNAL.value}:{self._source_asset}:{source_id}:{PIPELINE_VERSION}:{SCHEMA_VERSION}"
            node_id = hashlib.sha256(raw.encode()).hexdigest()
            label = str(row.get("label", row.get("name", row.get("description", source_id))))[:200]

            properties = {
                k: v for k, v in row.items()
                if k not in ("signal_id", "problem_signal_id", "id", "label", "name", "description", "confidence", "metadata", "evidence_ids")
            }
            properties.setdefault("description", str(row.get("description", "")))
            ev_ids = row.get("evidence_ids", [])
            if ev_ids:
                properties["evidence_ids"] = list(ev_ids) if hasattr(ev_ids, "__iter__") and not isinstance(ev_ids, str) else [str(ev_ids)]

            meta = {
                "source_asset": self._source_asset,
                "source_id": source_id,
                "run_id": "",
            }

            confidence = float(row.get("confidence", 1.0))

            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.PROBLEM_SIGNAL,
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
