from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _compute_id(prefix: str, *parts: str) -> str:
    raw = prefix + "-" + "-".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ReportType(str, Enum):
    startup_opportunity = "startup_opportunity"
    investor = "investor"
    company = "company"
    product = "product"
    technology_landscape = "technology_landscape"
    market_intelligence = "market_intelligence"
    competitor_analysis = "competitor_analysis"
    weekly = "weekly"
    monthly = "monthly"
    executive_summary = "executive_summary"


class ReportFormat(str, Enum):
    json = "json"
    markdown = "markdown"
    html = "html"
    csv = "csv"
    pdf = "pdf"
    docx = "docx"
    pptx = "pptx"


class SectionType(str, Enum):
    executive_summary = "executive_summary"
    top_findings = "top_findings"
    trend_analysis = "trend_analysis"
    opportunity_analysis = "opportunity_analysis"
    reasoning_summary = "reasoning_summary"
    root_causes = "root_causes"
    evidence = "evidence"
    confidence = "confidence"
    charts = "charts"
    recommendations = "recommendations"
    appendix = "appendix"


class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    pie = "pie"
    timeline = "timeline"
    heatmap = "heatmap"
    treemap = "treemap"
    sankey = "sankey"


class ComparisonChange(str, Enum):
    added = "added"
    removed = "removed"
    changed = "changed"
    unchanged = "unchanged"


class ReportVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    report_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    template_version: str = "1.0.0"
    evaluation_version: str = "1.0.0"


class SourceLineage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    knowledge_graph_run_id: str | None = None
    reasoning_run_id: str | None = None
    opportunity_run_id: str | None = None
    trend_run_id: str | None = None
    snapshot_id: str | None = None
    pipeline_manifest: dict[str, Any] = Field(default_factory=dict)


class ReportSummaries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    one_paragraph: str = ""
    five_bullets: list[str] = Field(default_factory=list)
    one_sentence: str = ""
    one_tweet: str = ""
    linkedin_summary: str = ""
    json_summary: dict[str, Any] = Field(default_factory=dict)


class SectionProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    reasoning_chain_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    kg_node_ids: list[str] = Field(default_factory=list)
    cluster_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    opportunity_ids: list[str] = Field(default_factory=list)
    trend_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SectionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_type: SectionType
    title: str
    order: int
    renderer_hints: dict[str, Any] = Field(default_factory=dict)


class ChartSeries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    values: list[float | int | str]
    color: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chart_id: str = ""
    chart_type: ChartType
    title: str
    description: str = ""
    series: list[ChartSeries] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "chart_id" not in data or not data["chart_id"]:
            raw = f"chart-{data.get('chart_type', 'unknown')}-{data.get('title', '')}-{time.time_ns()}"
            data["chart_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        super().__init__(**data)


class TableSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    table_id: str = ""
    title: str
    headers: list[str]
    rows: list[list[str | float | int]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "table_id" not in data or not data["table_id"]:
            raw = f"table-{data.get('title', '')}-{time.time_ns()}"
            data["table_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        super().__init__(**data)


class Highlight(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str
    source: str
    score: float
    section: SectionType


class ReportSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_id: str = ""
    section_type: SectionType
    title: str
    order: int
    content: dict[str, Any] = Field(default_factory=dict)
    charts: list[ChartSpec] = Field(default_factory=list)
    summaries: ReportSummaries = Field(default_factory=ReportSummaries)
    provenance: SectionProvenance = Field(default_factory=SectionProvenance)
    renderer_hints: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "section_id" not in data or not data["section_id"]:
            raw = f"section-{data.get('section_type', 'unknown')}-{data.get('title', '')}-{time.time_ns()}"
            data["section_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        super().__init__(**data)


class ReportAssets(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    charts: list[ChartSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    highlights: list[Highlight] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)


class PresentationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = ""
    report_type: ReportType
    title: str
    subtitle: str = ""
    generated_at: str = ""
    locale: str = "en-US"
    versions: ReportVersion = Field(default_factory=ReportVersion)
    lineage: SourceLineage = Field(default_factory=SourceLineage)
    sections: list[ReportSection] = Field(default_factory=list)
    assets: ReportAssets = Field(default_factory=ReportAssets)
    summaries: ReportSummaries = Field(default_factory=ReportSummaries)
    tags: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    checksum: str = ""

    def __init__(self, **data: Any) -> None:
        if "generated_at" not in data or not data["generated_at"]:
            data["generated_at"] = _now_iso()
        if "report_id" not in data or not data["report_id"]:
            raw = f"report-{data.get('report_type', 'unknown')}-{data.get('title', '')}-{time.time_ns()}"
            data["report_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        super().__init__(**data)


class TrendDelta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trend_id: str
    title: str
    change: ComparisonChange
    growth_pct_before: float = 0.0
    growth_pct_after: float = 0.0
    score_before: float = 0.0
    score_after: float = 0.0


class ReportComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    comparison_id: str = ""
    report_a_id: str
    report_b_id: str
    generated_at: str = ""
    new_opportunities: list[str] = Field(default_factory=list)
    removed_opportunities: list[str] = Field(default_factory=list)
    changed_opportunities: list[dict] = Field(default_factory=list)
    changed_trends: list[TrendDelta] = Field(default_factory=list)
    new_companies: list[str] = Field(default_factory=list)
    new_products: list[str] = Field(default_factory=list)
    confidence_deltas: dict[str, float] = Field(default_factory=dict)
    score_deltas: dict[str, float] = Field(default_factory=dict)
    summary: str = ""

    def __init__(self, **data: Any) -> None:
        if "generated_at" not in data or not data["generated_at"]:
            data["generated_at"] = _now_iso()
        if "comparison_id" not in data or not data["comparison_id"]:
            raw = f"comparison-{data.get('report_a_id', '')}-{data.get('report_b_id', '')}-{time.time_ns()}"
            data["comparison_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        super().__init__(**data)


class ReportIndexEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    report_id: str
    report_type: ReportType
    title: str
    generated_at: str
    tags: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    formats: list[ReportFormat] = Field(default_factory=list)
    sections: list[SectionType] = Field(default_factory=list)


class ReportIndex(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    entries: dict[str, ReportIndexEntry] = Field(default_factory=dict)
    by_tag: dict[str, list[str]] = Field(default_factory=dict)
    by_company: dict[str, list[str]] = Field(default_factory=dict)
    by_technology: dict[str, list[str]] = Field(default_factory=dict)
    by_product: dict[str, list[str]] = Field(default_factory=dict)
    by_opportunity: dict[str, list[str]] = Field(default_factory=dict)
    by_topic: dict[str, list[str]] = Field(default_factory=dict)
    by_date: dict[str, list[str]] = Field(default_factory=dict)
    by_type: dict[str, list[str]] = Field(default_factory=dict)


class ReportOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    report_id: str
    report_type: ReportType
    title: str
    generated_at: str
    sections_count: int
    charts_count: int
    formats: list[ReportFormat]
    checksums: dict[str, str] = Field(default_factory=dict)
    index_entry: ReportIndexEntry
    elapsed_seconds: float
