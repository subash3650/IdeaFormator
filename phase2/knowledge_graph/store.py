"""KnowledgeGraphStore — Parquet-based persistence for graph nodes and edges.

Always overwrites existing files. Writes empty Parquet with correct schema
when zero records. Embeds pipeline metadata via PyArrow schema metadata.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from pain_intelligence.knowledge.metadata import make_asset_metadata, read_parquet_metadata, write_parquet_with_metadata
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphMetadata, GraphNode, NodeType


def _derive_node_id(
    node_type: NodeType,
    source_asset: str,
    source_id: str,
    pipeline_version: str,
    schema_version: str,
) -> str:
    raw = f"{node_type.value}:{source_asset}:{source_id}:{pipeline_version}:{schema_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _derive_edge_id(
    edge_type: EdgeType,
    source_node_id: str,
    target_node_id: str,
    pipeline_version: str,
    schema_version: str,
    builder_version: str = "1.0",
) -> str:
    raw = f"{edge_type.value}:{source_node_id}:{target_node_id}:{pipeline_version}:{schema_version}:{builder_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


_NODE_SCHEMA: dict[str, pl.DataType] = {
    "node_id": pl.Utf8,
    "node_type": pl.Utf8,
    "label": pl.Utf8,
    "properties": pl.Utf8,
    "metadata": pl.Utf8,
    "attributes": pl.Utf8,
    "source_asset": pl.Utf8,
    "source_id": pl.Utf8,
    "confidence": pl.Float64,
    "created_at": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "schema_version": pl.Utf8,
}

_EDGE_SCHEMA: dict[str, pl.DataType] = {
    "edge_id": pl.Utf8,
    "source_node_id": pl.Utf8,
    "target_node_id": pl.Utf8,
    "edge_type": pl.Utf8,
    "weight": pl.Float64,
    "confidence": pl.Float64,
    "properties": pl.Utf8,
    "metadata": pl.Utf8,
    "attributes": pl.Utf8,
    "source_asset": pl.Utf8,
    "created_at": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "schema_version": pl.Utf8,
}


def _node_to_row(node: GraphNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type.value,
        "label": node.label,
        "properties": json.dumps(node.properties),
        "metadata": json.dumps(node.metadata),
        "attributes": json.dumps(node.attributes),
        "source_asset": node.source_asset,
        "source_id": node.source_id,
        "confidence": node.confidence,
        "created_at": node.created_at,
        "pipeline_version": node.pipeline_version,
        "schema_version": node.schema_version,
    }


def _row_to_node(row: dict[str, Any]) -> GraphNode:
    return GraphNode(
        node_id=row["node_id"],
        node_type=NodeType(row["node_type"]),
        label=row["label"],
        properties=json.loads(row.get("properties", "{}") or "{}"),
        metadata=json.loads(row.get("metadata", "{}") or "{}"),
        attributes=json.loads(row.get("attributes", "{}") or "{}"),
        source_asset=row["source_asset"],
        source_id=row["source_id"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        pipeline_version=row["pipeline_version"],
        schema_version=row["schema_version"],
    )


def _edge_to_row(edge: GraphEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "edge_type": edge.edge_type.value,
        "weight": edge.weight,
        "confidence": edge.confidence,
        "properties": json.dumps(edge.properties),
        "metadata": json.dumps(edge.metadata),
        "attributes": json.dumps(edge.attributes),
        "source_asset": edge.source_asset,
        "created_at": edge.created_at,
        "pipeline_version": edge.pipeline_version,
        "schema_version": edge.schema_version,
    }


def _row_to_edge(row: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        edge_id=row["edge_id"],
        source_node_id=row["source_node_id"],
        target_node_id=row["target_node_id"],
        edge_type=EdgeType(row["edge_type"]),
        weight=row["weight"],
        confidence=row["confidence"],
        properties=json.loads(row.get("properties", "{}") or "{}"),
        metadata=json.loads(row.get("metadata", "{}") or "{}"),
        attributes=json.loads(row.get("attributes", "{}") or "{}"),
        source_asset=row["source_asset"],
        created_at=row["created_at"],
        pipeline_version=row["pipeline_version"],
        schema_version=row["schema_version"],
    )


MANIFEST_FILENAME = "pipeline_manifest.json"


class KnowledgeGraphStore:
    """Persists graph nodes and edges as Parquet files.

    File layout:
        {base_path}/knowledge_graph/
            kg_nodes.parquet
            kg_edges.parquet
            kg_metadata.json
            kg_manifest.json
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._graph_dir = self._base_path / "knowledge_graph"
        self._graph_dir.mkdir(parents=True, exist_ok=True)

    @property
    def nodes_path(self) -> Path:
        return self._graph_dir / "kg_nodes.parquet"

    @property
    def edges_path(self) -> Path:
        return self._graph_dir / "kg_edges.parquet"

    @property
    def metadata_path(self) -> Path:
        return self._graph_dir / "kg_metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self._graph_dir / "kg_manifest.json"

    @property
    def report_path(self) -> Path:
        return self._graph_dir / "graph_report.json"

    @property
    def statistics_path(self) -> Path:
        return self._graph_dir / "graph_statistics.json"

    @property
    def dashboard_json_path(self) -> Path:
        return self._graph_dir / "graph_dashboard.json"

    @property
    def dashboard_txt_path(self) -> Path:
        return self._graph_dir / "graph_dashboard.txt"

    @property
    def summary_path(self) -> Path:
        return self._graph_dir / "graph_summary.md"

    # ── Node I/O ─────────────────────────────────────────────────────

    def save_nodes(self, nodes: list[GraphNode], run_id: str, input_checksum: str = "") -> Path:
        metadata = make_asset_metadata(
            run_id=run_id,
            input_checksum=input_checksum,
            record_count=len(nodes),
            asset="kg_nodes.parquet",
        )
        if not nodes:
            df = pl.DataFrame(schema=_NODE_SCHEMA)
        else:
            rows = [_node_to_row(n) for n in nodes]
            df = pl.DataFrame(rows, schema=_NODE_SCHEMA)
        return write_parquet_with_metadata(df, self.nodes_path, metadata)

    def load_nodes(self) -> list[GraphNode]:
        if not self.nodes_path.exists():
            return []
        df = pl.read_parquet(str(self.nodes_path))
        return [_row_to_node(row) for row in df.to_dicts()]

    def load_nodes_df(self) -> pl.DataFrame:
        if not self.nodes_path.exists():
            return pl.DataFrame(schema=_NODE_SCHEMA)
        return pl.read_parquet(str(self.nodes_path))

    # ── Edge I/O ─────────────────────────────────────────────────────

    def save_edges(self, edges: list[GraphEdge], run_id: str, input_checksum: str = "") -> Path:
        metadata = make_asset_metadata(
            run_id=run_id,
            input_checksum=input_checksum,
            record_count=len(edges),
            asset="kg_edges.parquet",
        )
        if not edges:
            df = pl.DataFrame(schema=_EDGE_SCHEMA)
        else:
            rows = [_edge_to_row(e) for e in edges]
            df = pl.DataFrame(rows, schema=_EDGE_SCHEMA)
        return write_parquet_with_metadata(df, self.edges_path, metadata)

    def load_edges(self) -> list[GraphEdge]:
        if not self.edges_path.exists():
            return []
        df = pl.read_parquet(str(self.edges_path))
        return [_row_to_edge(row) for row in df.to_dicts()]

    def load_edges_df(self) -> pl.DataFrame:
        if not self.edges_path.exists():
            return pl.DataFrame(schema=_EDGE_SCHEMA)
        return pl.read_parquet(str(self.edges_path))

    # ── Metadata I/O ─────────────────────────────────────────────────

    def save_metadata(self, metadata: GraphMetadata) -> Path:
        self.metadata_path.write_text(json.dumps(metadata.model_dump(mode="json"), indent=2, default=str), encoding="utf-8")
        return self.metadata_path

    def load_metadata(self) -> GraphMetadata | None:
        if not self.metadata_path.exists():
            return None
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return GraphMetadata(**data)

    # ── Manifest I/O ─────────────────────────────────────────────────

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return self.manifest_path

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    # ── Report / Export I/O ──────────────────────────────────────────

    def save_report(self, report: dict[str, Any]) -> Path:
        self.report_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return self.report_path

    def save_statistics(self, statistics: dict[str, Any]) -> Path:
        self.statistics_path.write_text(json.dumps(statistics, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return self.statistics_path

    def save_dashboard_json(self, dashboard: dict[str, Any]) -> Path:
        self.dashboard_json_path.write_text(json.dumps(dashboard, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return self.dashboard_json_path

    def save_dashboard_txt(self, text: str) -> Path:
        self.dashboard_txt_path.write_text(text, encoding="utf-8")
        return self.dashboard_txt_path

    def save_summary(self, text: str) -> Path:
        self.summary_path.write_text(text, encoding="utf-8")
        return self.summary_path

    # ── Status ─────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self.nodes_path.exists() and self.edges_path.exists()

    def count(self) -> dict[str, int]:
        return {
            "nodes": len(self.load_nodes()),
            "edges": len(self.load_edges()),
        }

    def checksums(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, path in [
            ("kg_nodes.parquet", self.nodes_path),
            ("kg_edges.parquet", self.edges_path),
            ("kg_metadata.json", self.metadata_path),
        ]:
            if path.exists():
                h = hashlib.sha256()
                h.update(path.read_bytes())
                result[name] = h.hexdigest()[:16]
        return result

    def run_id(self) -> str:
        meta = read_parquet_metadata(self.nodes_path)
        return meta.get("run_id", "")

    def get_graph_dir(self) -> Path:
        return self._graph_dir
