from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phase4.copilot.schema import ResponseFormat


class CopilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: Path = Field(default=Path("pain_intelligence/knowledge/assets/phase3"))
    knowledge_dir: Path | None = Field(default=None)

    enabled_tools: list[str] = Field(
        default_factory=lambda: [
            "knowledge_graph", "reasoning", "opportunity",
            "trend", "presentation", "search", "comparison", "evidence",
        ]
    )

    planner_max_steps: int = Field(default=5, ge=1, le=10)
    planner_timeout_ms: float = Field(default=500.0, ge=50, le=5000)
    planner_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    tool_timeout_ms: float = Field(default=2000.0, ge=100, le=10000)
    max_results_per_tool: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)

    max_short_term: int = Field(default=20, ge=5, le=100)
    max_conversation: int = Field(default=50, ge=10, le=200)
    max_long_term: int = Field(default=100, ge=10, le=500)
    max_pinned: int = Field(default=20, ge=5, le=100)
    compression_threshold: int = Field(default=100, ge=10, le=500)
    max_entity_history: int = Field(default=100, ge=10, le=500)
    session_ttl_minutes: int = Field(default=60, ge=5, le=1440)

    min_citation_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    max_citations_per_response: int = Field(default=10, ge=1, le=50)

    default_response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)
    max_response_length: int = Field(default=4000, ge=500, le=20000)
    enable_suggested_followups: bool = Field(default=True)
    enable_streaming: bool = Field(default=True)

    llm_provider: str = Field(default="mock")
    llm_model: str = Field(default="mock")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=100, le=8000)

    evaluation_enabled: bool = Field(default=True)

    @property
    def phase2_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base.parent / "phase2"

    @property
    def phase3_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base

    @property
    def report_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base / "reports"

    @property
    def copilot_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base / "copilot"


def load_copilot_config(path: str | Path | None = None) -> CopilotConfig:
    if path is None:
        path = Path("configs/default.yaml")
    path = Path(path)

    if not path.exists():
        return CopilotConfig()

    with open(str(path), encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = raw.get("copilot", {})

    if "output_dir" in cfg and isinstance(cfg["output_dir"], str):
        cfg["output_dir"] = Path(cfg["output_dir"])
    if "knowledge_dir" in cfg and isinstance(cfg["knowledge_dir"], str):
        cfg["knowledge_dir"] = Path(cfg["knowledge_dir"])

    if "default_response_format" in cfg and isinstance(cfg["default_response_format"], str):
        cfg["default_response_format"] = ResponseFormat(cfg["default_response_format"])

    return CopilotConfig(**cfg)
