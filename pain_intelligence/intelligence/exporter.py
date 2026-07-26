"""Export utilities for the Knowledge Extraction Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.intelligence.reports import build_reports
from pain_intelligence.intelligence.schema import Evidence, Observation, ProblemSignal
from pain_intelligence.knowledge.store import KnowledgeStore


class KnowledgeExporter:
    """Exports knowledge assets as Parquet + JSON.

    ALWAYS writes assets, even with zero records (writes empty Parquet).
    """

    def __init__(self, store: KnowledgeStore, output_dir: str | Path = "reports") -> None:
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._input_checksum: str = ""
        self._input_document_count: int = 0

    @property
    def input_checksum(self) -> str:
        return self._input_checksum

    @input_checksum.setter
    def input_checksum(self, value: str) -> None:
        self._input_checksum = value

    @property
    def input_document_count(self) -> int:
        return self._input_document_count

    @input_document_count.setter
    def input_document_count(self, value: int) -> None:
        self._input_document_count = value

    def export(
        self,
        observations: list[Observation] | None = None,
        evidence: list[Evidence] | None = None,
        signals: list[ProblemSignal] | None = None,
        filtering_stats: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Export all assets to Parquet (via store) and JSON (via reports dir).

        ALWAYS writes each asset — even if empty — to prevent stale data.
        """
        exported: dict[str, Path] = {}

        # Store assets (Parquet) — always write, even if empty
        obs_df = self._observations_to_df(observations or [])
        path = self.store.write_asset(
            "observations",
            obs_df,
            input_checksum=self._input_checksum,
            input_document_count=self._input_document_count,
        )
        exported["observations.parquet"] = path

        ev_df = Evidence.to_dataframe(evidence or [])
        path = self.store.write_asset(
            "evidence",
            ev_df,
            input_checksum=self._input_checksum,
            input_document_count=self._input_document_count,
        )
        exported["evidence.parquet"] = path

        sig_df = ProblemSignal.to_dataframe(signals or [])
        path = self.store.write_asset(
            "problem_signals",
            sig_df,
            input_checksum=self._input_checksum,
            input_document_count=self._input_document_count,
        )
        exported["problem_signals.parquet"] = path

        # Comprehensive intelligence report (JSON)
        reports = build_reports(
            observations=observations,
            evidence=evidence,
            signals=signals,
            filtering_stats=filtering_stats,
        )
        json_path = self.output_dir / "intelligence_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str, ensure_ascii=False)
        exported["intelligence_report.json"] = json_path

        # Signal quality report
        if filtering_stats is not None or signals is not None:
            quality = build_reports(
                observations=observations,
                evidence=evidence,
                signals=signals,
                filtering_stats=filtering_stats,
            ).get("signal_quality", {})
            quality_path = self.output_dir / "signal_quality_report.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(quality, f, indent=2, default=str, ensure_ascii=False)
            exported["signal_quality_report.json"] = quality_path

        return exported

    @staticmethod
    def _observations_to_df(observations: list[Observation]) -> pl.DataFrame:
        records = []
        for o in observations:
            d = o.model_dump()
            d["type"] = d["type"].value if d.get("type") else None
            d["method"] = d["method"].value if d.get("method") else None
            d["entity_type"] = d["entity_type"].value if d.get("entity_type") else None
            d["category"] = d.get("category") if d.get("category") else None
            records.append(d)
        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records, schema_overrides={
            "entity_type": pl.String,
            "category": pl.String,
            "canonical_value": pl.String,
            "canonical_source": pl.String,
            "pattern_label": pl.String,
        })
