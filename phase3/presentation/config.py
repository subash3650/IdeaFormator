from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phase3.presentation.schema import ReportFormat, ReportType


class PresentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: Path = Field(default=Path("pain_intelligence/knowledge/assets/phase3"))
    knowledge_dir: Path | None = Field(default=None)
    reports_dir: Path = Field(default=Path("reports"))

    enabled_report_types: list[ReportType] = Field(default_factory=lambda: list(ReportType))
    enabled_formats: list[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.json, ReportFormat.markdown, ReportFormat.html, ReportFormat.csv]
    )
    default_format: ReportFormat = Field(default=ReportFormat.json)

    max_findings: int = Field(default=20, ge=5, le=100)
    max_charts: int = Field(default=15, ge=3, le=50)
    max_trends_displayed: int = Field(default=15, ge=5, le=50)
    max_opportunities_displayed: int = Field(default=15, ge=5, le=50)
    max_root_causes_displayed: int = Field(default=10, ge=3, le=30)
    max_evidence_displayed: int = Field(default=20, ge=5, le=50)
    max_appendix_items: int = Field(default=50, ge=10, le=200)

    default_template: str = Field(default="executive")
    enabled_templates: list[str] = Field(
        default_factory=lambda: ["executive", "investor", "founder", "market", "technology"]
    )

    chart_width: int = Field(default=800, ge=400, le=1920)
    chart_height: int = Field(default=500, ge=300, le=1080)
    chart_theme: str = Field(default="plotly_white")

    html_title: str = Field(default="IdeaFormator Intelligence Report")
    html_author: str = Field(default="IdeaFormator Pipeline")

    enable_scheduler: bool = Field(default=False)
    schedules: list[dict[str, Any]] = Field(default_factory=list)

    evaluation_enabled: bool = Field(default=True)

    @field_validator("output_dir", "knowledge_dir", "reports_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Any:
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def report_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base / "reports"

    @property
    def phase2_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base.parent / "phase2"

    @property
    def phase3_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base


def load_presentation_config(path: str | Path | None = None) -> PresentationConfig:
    if path is None:
        path = Path("configs/default.yaml")
    path = Path(path)

    if not path.exists():
        return PresentationConfig()

    with open(str(path), encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = raw.get("presentation", {})

    if "output_dir" in cfg and isinstance(cfg["output_dir"], str):
        cfg["output_dir"] = Path(cfg["output_dir"])
    if "knowledge_dir" in cfg and isinstance(cfg["knowledge_dir"], str):
        cfg["knowledge_dir"] = Path(cfg["knowledge_dir"])
    if "reports_dir" in cfg and isinstance(cfg["reports_dir"], str):
        cfg["reports_dir"] = Path(cfg["reports_dir"])

    if "enabled_report_types" in cfg:
        cfg["enabled_report_types"] = [ReportType(t) for t in cfg["enabled_report_types"]]
    if "enabled_formats" in cfg:
        cfg["enabled_formats"] = [ReportFormat(f) for f in cfg["enabled_formats"]]
    if "default_format" in cfg:
        cfg["default_format"] = ReportFormat(cfg["default_format"])

    return PresentationConfig(**cfg)
