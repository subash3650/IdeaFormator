"""CausalEdgeBuilder — builds CAUSES and SUPPORTED_BY edges between signals and evidence."""

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


@register_edge_builder("causal")
class CausalEdgeBuilder(EdgeBuilder):
    """Builds CAUSES / SUPPORTED_BY edges from problem_signals.parquet.

    Creates edges linking EVIDENCE nodes to PROBLEM_SIGNAL nodes
    based on the evidence_ids listed in each signal.
    """

    def __init__(self, signal_asset_path: Path, config: Any = None) -> None:
        self._signal_path = Path(signal_asset_path)
        self._config = config
        self._source_asset = "problem_signals.parquet"
        self._min_confidence = getattr(config, "minimum_confidence", 0.5) if config else 0.5

    def build_edges(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        if not self._signal_path.exists():
            return []
        df = pl.read_parquet(str(self._signal_path))
        if df.height == 0:
            return []

        node_by_source_id: dict[str, str] = {}
        for node in nodes:
            node_by_source_id[node.source_id] = node.node_id

        signal_nodes: dict[str, str] = {}
        evidence_nodes: dict[str, str] = {}
        for node in nodes:
            if node.node_type.value == "problem_signal":
                signal_nodes[node.source_id] = node.node_id
            elif node.node_type.value == "evidence":
                evidence_nodes[node.source_id] = node.node_id

        edges: list[GraphEdge] = []
        now = datetime.now(timezone.utc).isoformat()

        for row in df.to_dicts():
            signal_source_id = str(row.get("signal_id", row.get("problem_signal_id", row.get("id", ""))))
            signal_nid = signal_nodes.get(signal_source_id) or node_by_source_id.get(signal_source_id)
            if signal_nid is None:
                continue

            evidence_ids_raw = row.get("evidence_ids", [])
            if hasattr(evidence_ids_raw, "__iter__") and not isinstance(evidence_ids_raw, str):
                ev_ids: list[str] = list(evidence_ids_raw)
            else:
                ev_ids = [str(evidence_ids_raw)] if evidence_ids_raw else []

            for ev_source_id in ev_ids:
                ev_nid = evidence_nodes.get(ev_source_id) or node_by_source_id.get(ev_source_id)
                if ev_nid is None:
                    continue

                raw = f"{EdgeType.CAUSES.value}:{ev_nid}:{signal_nid}:{PIPELINE_VERSION}:{SCHEMA_VERSION}:1.0"
                edge_id = hashlib.sha256(raw.encode()).hexdigest()
                confidence = float(row.get("confidence", 0.8))

                edge = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=ev_nid,
                    target_node_id=signal_nid,
                    edge_type=EdgeType.CAUSES,
                    weight=confidence,
                    confidence=confidence,
                    properties={},
                    metadata={
                        "source_asset": self._source_asset,
                        "signal_id": signal_source_id,
                        "evidence_id": ev_source_id,
                    },
                    attributes={},
                    source_asset=self._source_asset,
                    created_at=now,
                    pipeline_version=PIPELINE_VERSION,
                    schema_version=SCHEMA_VERSION,
                )
                edges.append(edge)

        return edges
