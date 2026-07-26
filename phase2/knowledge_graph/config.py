"""Configuration models for the Knowledge Graph Infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from phase2.knowledge_graph.schema import EdgeType, NodeType


class KnowledgeGraphConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    output_dir: Path = Field(description="Output directory for knowledge graph assets")
    knowledge_dir: Path | None = Field(None, description="Override output directory for graph files")

    node_types: list[NodeType] = Field(
        default=[
            NodeType.DOCUMENT,
            NodeType.OBSERVATION,
            NodeType.ENTITY,
            NodeType.EVIDENCE,
            NodeType.PROBLEM_SIGNAL,
            NodeType.CLUSTER,
        ],
        description="Node types to include in the graph",
    )
    edge_types: list[EdgeType] = Field(
        default=[
            EdgeType.SIMILAR_TO,
            EdgeType.BELONGS_TO,
            EdgeType.CAUSES,
            EdgeType.DERIVED_FROM,
            EdgeType.MENTIONS,
        ],
        description="Edge types to include in the graph",
    )

    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for inclusion")
    minimum_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum edge weight for inclusion")

    include_isolated_nodes: bool = Field(default=True, description="Include nodes with degree zero")
    deterministic: bool = Field(default=True, description="Enforce deterministic outputs")

    version: str = Field(default="1.0", description="Engine version")
    builder_version: str = Field(default="1.0", description="Builder version")

    @property
    def graph_dir(self) -> Path:
        return (self.knowledge_dir or self.output_dir) / "knowledge_graph"


def load_knowledge_graph_config(path: str | Path) -> KnowledgeGraphConfig:
    """Load knowledge graph config from YAML file, falling back to defaults."""
    path = Path(path)
    if not path.exists():
        return KnowledgeGraphConfig(output_dir=Path("pain_intelligence/knowledge/assets/phase2"))

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    kg_cfg = raw.get("knowledge_graph", {})
    output_dir = kg_cfg.get("output_dir") or raw.get("output_directory")
    if output_dir is not None:
        kg_cfg["output_dir"] = Path(output_dir)

    knowledge_dir = kg_cfg.get("knowledge_dir")
    if knowledge_dir is not None:
        kg_cfg["knowledge_dir"] = Path(knowledge_dir)

    node_types_raw = kg_cfg.pop("node_types", None)
    if node_types_raw is not None:
        kg_cfg["node_types"] = [NodeType(v) for v in node_types_raw]

    edge_types_raw = kg_cfg.pop("edge_types", None)
    if edge_types_raw is not None:
        kg_cfg["edge_types"] = [EdgeType(v) for v in edge_types_raw]

    return KnowledgeGraphConfig(**kg_cfg)
