"""Configuration for the Knowledge Graph Reasoning Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from phase2.reasoning.schema import ExplanationFormat, PropagationStrategy, RootCauseRanking


class ReasoningConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    output_dir: Path
    knowledge_dir: Path | None = Field(default=None)

    enabled_rules: list[str] = Field(
        default_factory=lambda: [
            "transitive_closure",
            "causal_chain",
            "evidence_convergence",
        ],
    )
    enabled_reasoning_types: list[str] = Field(
        default_factory=lambda: ["transitive", "causal_chain", "evidence_aggregation"],
    )

    propagation_strategy: PropagationStrategy = Field(default=PropagationStrategy.MULTIPLICATIVE)
    decay_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    max_chain_length: int = Field(default=8, ge=1, le=50)
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    max_inferences_per_run: int = Field(default=10000, ge=1)
    max_rule_iterations: int = Field(default=5, ge=1, le=20)

    root_cause_ranking: RootCauseRanking = Field(default=RootCauseRanking.TRANSITIVE_IMPACT)
    max_root_cause_depth: int = Field(default=8, ge=1, le=50)

    min_evidence_count: int = Field(default=2, ge=1)
    conflicting_evidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    generate_explanations: bool = Field(default=True)
    explanation_format: ExplanationFormat = Field(default=ExplanationFormat.TEMPLATE)
    collapse_chains_longer_than: int = Field(default=4, ge=1)

    cache_enabled: bool = Field(default=True)

    version: str = Field(default="1.0")
    reasoning_version: str = Field(default="1.0")

    @property
    def reasoning_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base / "reasoning"


def load_reasoning_config(path: str | Path) -> ReasoningConfig:
    path = Path(path)
    default_output = Path("pain_intelligence/knowledge/assets/phase2")
    if not path.exists():
        return ReasoningConfig(output_dir=default_output)
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    rc = raw.get("reasoning", {})
    output_dir = rc.get("output_dir") or raw.get("output_directory")
    if output_dir is not None:
        rc["output_dir"] = Path(output_dir)
    else:
        rc["output_dir"] = default_output
    knowledge_dir = rc.get("knowledge_dir")
    if knowledge_dir is not None:
        rc["knowledge_dir"] = Path(knowledge_dir)
    for enum_field, enum_cls in [
        ("propagation_strategy", PropagationStrategy),
        ("root_cause_ranking", RootCauseRanking),
        ("explanation_format", ExplanationFormat),
    ]:
        val = rc.pop(enum_field, None)
        if val is not None:
            rc[enum_field] = enum_cls(val)
    return ReasoningConfig(**rc)
