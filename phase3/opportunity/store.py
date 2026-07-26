"""OpportunityStore — Parquet persistence for discovered opportunities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    MarketMaturity,
    MarketSize,
    Opportunity,
    OpportunityMetadata,
    OpportunityStatus,
    OpportunityType,
    RecommendationType,
    ScoringBreakdown,
)

# ---------------------------------------------------------------------------
# Parquet schema definitions
# ---------------------------------------------------------------------------

OPPORTUNITY_SCHEMA: dict[str, pl.DataType] = {
    "opportunity_id": pl.Utf8,
    "title": pl.Utf8,
    "summary": pl.Utf8,
    "root_problem": pl.Utf8,
    "supporting_evidence": pl.Utf8,
    "reasoning_chain_ids": pl.Utf8,
    "cluster_ids": pl.Utf8,
    "kg_node_ids": pl.Utf8,
    "affected_products": pl.Utf8,
    "affected_companies": pl.Utf8,
    "affected_technologies": pl.Utf8,
    "estimated_market_size": pl.Utf8,
    "market_maturity": pl.Utf8,
    "implementation_complexity": pl.Utf8,
    "time_to_market": pl.Utf8,
    "investment_required": pl.Utf8,
    "estimated_revenue": pl.Utf8,
    "recurring_revenue_potential": pl.Boolean,
    "strategic_fit": pl.Utf8,
    "pain_severity": pl.Float64,
    "frequency_score": pl.Float64,
    "trend_score": pl.Float64,
    "competition_score": pl.Float64,
    "feasibility_score": pl.Float64,
    "opportunity_score": pl.Float64,
    "scoring_breakdown": pl.Utf8,
    "confidence": pl.Utf8,
    "recommendation_type": pl.Utf8,
    "suggested_solution": pl.Utf8,
    "suggested_business_model": pl.Utf8,
    "status": pl.Utf8,
    "rank": pl.Int64,
    "created_at": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "schema_version": pl.Utf8,
}

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _opportunity_to_row(opp: Opportunity) -> dict[str, Any]:
    return {
        "opportunity_id": opp.opportunity_id,
        "title": opp.title,
        "summary": opp.summary,
        "root_problem": opp.root_problem,
        "supporting_evidence": json.dumps(opp.supporting_evidence),
        "reasoning_chain_ids": json.dumps(opp.reasoning_chain_ids),
        "cluster_ids": json.dumps(opp.cluster_ids),
        "kg_node_ids": json.dumps(opp.kg_node_ids),
        "affected_products": json.dumps(opp.affected_products),
        "affected_companies": json.dumps(opp.affected_companies),
        "affected_technologies": json.dumps(opp.affected_technologies),
        "estimated_market_size": opp.estimated_market_size.value,
        "market_maturity": opp.market_maturity.value,
        "implementation_complexity": opp.implementation_complexity.value,
        "time_to_market": opp.time_to_market,
        "investment_required": opp.investment_required,
        "estimated_revenue": opp.estimated_revenue,
        "recurring_revenue_potential": opp.recurring_revenue_potential,
        "strategic_fit": opp.strategic_fit,
        "pain_severity": opp.pain_severity,
        "frequency_score": opp.frequency_score,
        "trend_score": opp.trend_score,
        "competition_score": opp.competition_score,
        "feasibility_score": opp.feasibility_score,
        "opportunity_score": opp.opportunity_score,
        "scoring_breakdown": json.dumps(opp.scoring_breakdown.model_dump(mode="json")),
        "confidence": json.dumps(opp.confidence.model_dump(mode="json")),
        "recommendation_type": opp.recommendation_type.value,
        "suggested_solution": opp.suggested_solution,
        "suggested_business_model": opp.suggested_business_model.value,
        "status": opp.status.value,
        "rank": opp.rank,
        "created_at": opp.created_at,
        "pipeline_version": opp.pipeline_version,
        "schema_version": opp.schema_version,
    }


def _row_to_opportunity(row: dict[str, Any]) -> Opportunity:
    sb_data = json.loads(row.get("scoring_breakdown") or "{}")
    conf_data = json.loads(row.get("confidence") or "{}")
    return Opportunity(
        opportunity_id=row["opportunity_id"],
        title=row.get("title", ""),
        summary=row.get("summary", ""),
        root_problem=row.get("root_problem", ""),
        supporting_evidence=json.loads(row.get("supporting_evidence") or "[]"),
        reasoning_chain_ids=json.loads(row.get("reasoning_chain_ids") or "[]"),
        cluster_ids=json.loads(row.get("cluster_ids") or "[]"),
        kg_node_ids=json.loads(row.get("kg_node_ids") or "[]"),
        affected_products=json.loads(row.get("affected_products") or "[]"),
        affected_companies=json.loads(row.get("affected_companies") or "[]"),
        affected_technologies=json.loads(row.get("affected_technologies") or "[]"),
        estimated_market_size=MarketSize(row.get("estimated_market_size", "unknown")),
        market_maturity=MarketMaturity(row.get("market_maturity", "unknown")),
        implementation_complexity=_str_to_impl_complexity(row.get("implementation_complexity", "")),
        time_to_market=row.get("time_to_market", ""),
        investment_required=row.get("investment_required", ""),
        estimated_revenue=row.get("estimated_revenue", ""),
        recurring_revenue_potential=bool(row.get("recurring_revenue_potential", False)),
        strategic_fit=row.get("strategic_fit", ""),
        pain_severity=float(row.get("pain_severity", 0)),
        frequency_score=float(row.get("frequency_score", 0)),
        trend_score=float(row.get("trend_score", 0)),
        competition_score=float(row.get("competition_score", 0)),
        feasibility_score=float(row.get("feasibility_score", 0)),
        opportunity_score=float(row.get("opportunity_score", 0)),
        scoring_breakdown=ScoringBreakdown(**sb_data),
        confidence=ConfidenceBreakdown(**conf_data),
        recommendation_type=RecommendationType(row.get("recommendation_type", "insufficient_data")),
        suggested_solution=row.get("suggested_solution", ""),
        suggested_business_model=OpportunityType(row.get("suggested_business_model", "saas")),
        status=OpportunityStatus(row.get("status", "identified")),
        rank=int(row.get("rank", 0)),
        created_at=row.get("created_at", ""),
        pipeline_version=row.get("pipeline_version", "1.0"),
        schema_version=row.get("schema_version", "1.0"),
    )


def _str_to_impl_complexity(val: str) -> Any:
    from phase3.opportunity.schema import ImplementationComplexity

    try:
        return ImplementationComplexity(val)
    except ValueError:
        return ImplementationComplexity.UNKNOWN


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class OpportunityStore:
    """Persists opportunities as Parquet files.

    File layout:
        {base_path}/opportunity/
            opportunities.parquet
            opportunity_metadata.json
            opportunity_manifest.json
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self._opp_dir = self._base_path / "opportunity"
        self._opp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def opportunity_dir(self) -> Path:
        return self._opp_dir

    @property
    def opportunities_path(self) -> Path:
        return self._opp_dir / "opportunities.parquet"

    @property
    def metadata_path(self) -> Path:
        return self._opp_dir / "opportunity_metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self._opp_dir / "opportunity_manifest.json"

    # -- Opportunities --

    def save_opportunities(
        self, opportunities: list[Opportunity], run_id: str
    ) -> Path:
        if not opportunities:
            df = pl.DataFrame(schema=OPPORTUNITY_SCHEMA)
        else:
            rows = [_opportunity_to_row(opp) for opp in opportunities]
            df = pl.DataFrame(rows, schema=OPPORTUNITY_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": str(len(opportunities)),
            "asset": "opportunities.parquet",
        }
        return write_parquet_with_metadata(df, self.opportunities_path, metadata=metadata)

    def load_opportunities(self) -> list[Opportunity]:
        if not self.opportunities_path.exists():
            return []
        df = pl.read_parquet(self.opportunities_path)
        return [_row_to_opportunity(row) for row in df.iter_rows(named=True)]

    # -- Metadata --

    def save_metadata(self, metadata: OpportunityMetadata) -> Path:
        self.metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        return self.metadata_path

    def load_metadata(self) -> OpportunityMetadata | None:
        if not self.metadata_path.exists():
            return None
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return OpportunityMetadata(**data)

    # -- Manifest --

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return self.manifest_path

    # -- Checksums --

    def checksums(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.opportunities_path.exists():
            h = hashlib.sha256()
            h.update(self.opportunities_path.read_bytes())
            result["opportunities.parquet"] = h.hexdigest()[:16]
        return result
