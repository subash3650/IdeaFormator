"""Reasoning store — Parquet persistence for derived knowledge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase2.reasoning.schema import (
    EvidenceAggregation,
    Explanation,
    InferenceResult,
    ProvenanceVersion,
    ReasoningChain,
    ReasoningMetadata,
    ReasoningStep,
)


def _serialize_value(val: Any) -> str:
    if isinstance(val, list):
        return json.dumps(val)
    if isinstance(val, dict) and not isinstance(val, (ProvenanceVersion,)):
        return json.dumps(val)
    return str(val) if val is not None else ""


def _serialize_provenance_version(pv: ProvenanceVersion) -> str:
    return json.dumps(pv.model_dump(mode="json"))


INFERENCE_SCHEMA: dict[str, pl.DataType] = {
    "inference_id": pl.Utf8,
    "inference_type": pl.Utf8,
    "derived_node_id": pl.Utf8,
    "derived_edge_id": pl.Utf8,
    "confidence": pl.Float64,
    "chain_id": pl.Utf8,
    "provenance": pl.Utf8,
    "created_at": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "schema_version": pl.Utf8,
}

CHAIN_SCHEMA: dict[str, pl.DataType] = {
    "chain_id": pl.Utf8,
    "inference_id": pl.Utf8,
    "steps": pl.Utf8,
    "input_node_ids": pl.Utf8,
    "output_node_ids": pl.Utf8,
    "output_edge_ids": pl.Utf8,
    "total_confidence": pl.Float64,
    "provenance_version": pl.Utf8,
    "created_at": pl.Utf8,
}

ROOT_CAUSE_SCHEMA: dict[str, pl.DataType] = {
    "cause_node_id": pl.Utf8,
    "cause_label": pl.Utf8,
    "effect_node_id": pl.Utf8,
    "effect_label": pl.Utf8,
    "path": pl.Utf8,
    "path_length": pl.Int64,
    "propagated_confidence": pl.Float64,
    "transitive_impact_count": pl.Int64,
    "evidence_count": pl.Int64,
    "ranking_score": pl.Float64,
    "ranking_method": pl.Utf8,
}

EVIDENCE_SCHEMA: dict[str, pl.DataType] = {
    "conclusion_node_id": pl.Utf8,
    "conclusion_label": pl.Utf8,
    "evidence_node_ids": pl.Utf8,
    "evidence_count": pl.Int64,
    "aggregated_confidence": pl.Float64,
    "aggregation_method": pl.Utf8,
    "conflicting_evidence_count": pl.Int64,
    "created_at": pl.Utf8,
}


class ReasoningStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self._reasoning_dir = self._base_path / "reasoning"
        self._reasoning_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def reasoning_dir(self) -> Path:
        return self._reasoning_dir

    @property
    def inferences_path(self) -> Path:
        return self._reasoning_dir / "derived_inferences.parquet"

    @property
    def chains_path(self) -> Path:
        return self._reasoning_dir / "reasoning_chains.parquet"

    @property
    def root_causes_path(self) -> Path:
        return self._reasoning_dir / "root_causes.parquet"

    @property
    def evidence_path(self) -> Path:
        return self._reasoning_dir / "evidence_aggregations.parquet"

    @property
    def explanations_path(self) -> Path:
        return self._reasoning_dir / "explanations.json"

    @property
    def metadata_path(self) -> Path:
        return self._reasoning_dir / "reasoning_metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self._reasoning_dir / "reasoning_manifest.json"

    # ── Inferences ─────────────────────────────────────────────────

    def save_inferences(self, inferences: list[InferenceResult], run_id: str) -> Path:
        if not inferences:
            df = pl.DataFrame(schema=INFERENCE_SCHEMA)
        else:
            rows = [_inference_to_row(inf) for inf in inferences]
            df = pl.DataFrame(rows, schema=INFERENCE_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": len(inferences),
            "asset": "derived_inferences.parquet",
        }
        return write_parquet_with_metadata(df, self.inferences_path, metadata=metadata)

    def load_inferences(self) -> list[InferenceResult]:
        if not self.inferences_path.exists():
            return []
        df = pl.read_parquet(self.inferences_path)
        return [_row_to_inference(row) for row in df.iter_rows(named=True)]

    # ── Chains ─────────────────────────────────────────────────────

    def save_chains(self, chains: list[ReasoningChain], run_id: str) -> Path:
        if not chains:
            df = pl.DataFrame(schema=CHAIN_SCHEMA)
        else:
            rows = [_chain_to_row(c) for c in chains]
            df = pl.DataFrame(rows, schema=CHAIN_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": len(chains),
            "asset": "reasoning_chains.parquet",
        }
        return write_parquet_with_metadata(df, self.chains_path, metadata=metadata)

    def load_chains(self) -> list[ReasoningChain]:
        if not self.chains_path.exists():
            return []
        df = pl.read_parquet(self.chains_path)
        return [_row_to_chain(row) for row in df.iter_rows(named=True)]

    # ── Root Causes ────────────────────────────────────────────────

    def save_root_causes(self, causes: list, run_id: str) -> Path:
        if not causes:
            df = pl.DataFrame(schema=ROOT_CAUSE_SCHEMA)
        else:
            rows = [_root_cause_to_row(c) for c in causes]
            df = pl.DataFrame(rows, schema=ROOT_CAUSE_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": len(causes),
            "asset": "root_causes.parquet",
        }
        return write_parquet_with_metadata(df, self.root_causes_path, metadata=metadata)

    def load_root_causes(self) -> list:
        if not self.root_causes_path.exists():
            return []
        df = pl.read_parquet(self.root_causes_path)
        return [_row_to_root_cause(row) for row in df.iter_rows(named=True)]

    # ── Evidence Aggregations ──────────────────────────────────────

    def save_evidence_aggregations(self, aggregations: list, run_id: str) -> Path:
        if not aggregations:
            df = pl.DataFrame(schema=EVIDENCE_SCHEMA)
        else:
            rows = [_evidence_to_row(a) for a in aggregations]
            df = pl.DataFrame(rows, schema=EVIDENCE_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": len(aggregations),
            "asset": "evidence_aggregations.parquet",
        }
        return write_parquet_with_metadata(df, self.evidence_path, metadata=metadata)

    def load_evidence_aggregations(self) -> list:
        if not self.evidence_path.exists():
            return []
        df = pl.read_parquet(self.evidence_path)
        return [_row_to_evidence(row) for row in df.iter_rows(named=True)]

    # ── Explanations ───────────────────────────────────────────────

    def save_explanations(self, explanations: list[Explanation]) -> Path:
        data = [e.model_dump(mode="json") for e in explanations]
        self.explanations_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        return self.explanations_path

    # ── Metadata ───────────────────────────────────────────────────

    def save_metadata(self, metadata: ReasoningMetadata) -> Path:
        self.metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        return self.metadata_path

    def load_metadata(self) -> ReasoningMetadata | None:
        if not self.metadata_path.exists():
            return None
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return ReasoningMetadata(**data)

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return self.manifest_path

    def checksums(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, path in [
            ("derived_inferences.parquet", self.inferences_path),
            ("reasoning_chains.parquet", self.chains_path),
            ("root_causes.parquet", self.root_causes_path),
            ("evidence_aggregations.parquet", self.evidence_path),
        ]:
            if path.exists():
                h = hashlib.sha256()
                h.update(path.read_bytes())
                result[name] = h.hexdigest()[:16]
        return result


# ── Row Serialization ────────────────────────────────────────────────────────

def _inference_to_row(inf: InferenceResult) -> dict[str, Any]:
    return {
        "inference_id": inf.inference_id,
        "inference_type": inf.inference_type.value,
        "derived_node_id": inf.derived_node_id or "",
        "derived_edge_id": inf.derived_edge_id or "",
        "confidence": inf.confidence,
        "chain_id": inf.chain_id,
        "provenance": json.dumps(inf.provenance),
        "created_at": inf.created_at,
        "pipeline_version": inf.pipeline_version,
        "schema_version": inf.schema_version,
    }


def _row_to_inference(row: dict[str, Any]) -> InferenceResult:
    prov = json.loads(row.get("provenance") or "[]")
    return InferenceResult(
        inference_id=row["inference_id"],
        inference_type=row["inference_type"],
        derived_node_id=row.get("derived_node_id") or None,
        derived_edge_id=row.get("derived_edge_id") or None,
        confidence=float(row.get("confidence", 0)),
        chain_id=row.get("chain_id", ""),
        provenance=prov,
        created_at=row.get("created_at", ""),
        pipeline_version=row.get("pipeline_version", "1.0"),
        schema_version=row.get("schema_version", "1.0"),
    )


def _chain_to_row(chain: ReasoningChain) -> dict[str, Any]:
    steps_data = [
        {
            "step_id": s.step_id,
            "rule_name": s.rule_name,
            "rule_version": s.rule_version,
            "input_node_ids": s.input_node_ids,
            "input_edge_ids": s.input_edge_ids,
            "output_node_id": s.output_node_id,
            "output_edge_id": s.output_edge_id,
            "confidence_delta": s.confidence_delta,
            "timestamp": s.timestamp,
        }
        for s in chain.steps
    ]
    return {
        "chain_id": chain.chain_id,
        "inference_id": chain.inference_id,
        "steps": json.dumps(steps_data),
        "input_node_ids": json.dumps(chain.input_node_ids),
        "output_node_ids": json.dumps(chain.output_node_ids),
        "output_edge_ids": json.dumps(chain.output_edge_ids),
        "total_confidence": chain.total_confidence,
        "provenance_version": json.dumps(chain.provenance_version.model_dump(mode="json")),
        "created_at": chain.created_at,
    }


def _row_to_chain(row: dict[str, Any]) -> ReasoningChain:
    steps_data = json.loads(row.get("steps") or "[]")
    steps = [
        ReasoningStep(
            step_id=s["step_id"],
            rule_name=s["rule_name"],
            rule_version=s.get("rule_version", "1.0"),
            input_node_ids=s.get("input_node_ids", []),
            input_edge_ids=s.get("input_edge_ids", []),
            output_node_id=s.get("output_node_id"),
            output_edge_id=s.get("output_edge_id"),
            confidence_delta=float(s.get("confidence_delta", 0)),
            timestamp=s.get("timestamp", ""),
        )
        for s in steps_data
    ]
    pv_data = json.loads(row.get("provenance_version") or "{}")
    return ReasoningChain(
        chain_id=row["chain_id"],
        inference_id=row.get("inference_id", ""),
        steps=steps,
        input_node_ids=json.loads(row.get("input_node_ids") or "[]"),
        output_node_ids=json.loads(row.get("output_node_ids") or "[]"),
        output_edge_ids=json.loads(row.get("output_edge_ids") or "[]"),
        total_confidence=float(row.get("total_confidence", 0)),
        provenance_version=ProvenanceVersion(**pv_data),
        created_at=row.get("created_at", ""),
    )


def _root_cause_to_row(rc) -> dict[str, Any]:
    return {
        "cause_node_id": rc.cause_node_id,
        "cause_label": rc.cause_label,
        "effect_node_id": rc.effect_node_id,
        "effect_label": rc.effect_label,
        "path": json.dumps(rc.path),
        "path_length": rc.path_length,
        "propagated_confidence": rc.propagated_confidence,
        "transitive_impact_count": rc.transitive_impact_count,
        "evidence_count": rc.evidence_count,
        "ranking_score": rc.ranking_score,
        "ranking_method": rc.ranking_method.value if rc.ranking_method else "",
    }


def _row_to_root_cause(row: dict[str, Any]):
    from phase2.reasoning.schema import RootCause, RootCauseRanking
    return RootCause(
        cause_node_id=row["cause_node_id"],
        cause_label=row.get("cause_label", ""),
        effect_node_id=row["effect_node_id"],
        effect_label=row.get("effect_label", ""),
        path=json.loads(row.get("path") or "[]"),
        path_length=int(row.get("path_length", 0)),
        propagated_confidence=float(row.get("propagated_confidence", 0)),
        transitive_impact_count=int(row.get("transitive_impact_count", 0)),
        evidence_count=int(row.get("evidence_count", 0)),
        ranking_score=float(row.get("ranking_score", 0)),
        ranking_method=RootCauseRanking(row["ranking_method"]) if row.get("ranking_method") else RootCauseRanking.TRANSITIVE_IMPACT,
    )


def _evidence_to_row(agg) -> dict[str, Any]:
    return {
        "conclusion_node_id": agg.conclusion_node_id,
        "conclusion_label": agg.conclusion_label,
        "evidence_node_ids": json.dumps(agg.evidence_node_ids),
        "evidence_count": agg.evidence_count,
        "aggregated_confidence": agg.aggregated_confidence,
        "aggregation_method": agg.aggregation_method,
        "conflicting_evidence_count": agg.conflicting_evidence_count,
        "created_at": agg.created_at,
    }


def _row_to_evidence(row: dict[str, Any]):
    return EvidenceAggregation(
        conclusion_node_id=row["conclusion_node_id"],
        conclusion_label=row.get("conclusion_label", ""),
        evidence_node_ids=json.loads(row.get("evidence_node_ids") or "[]"),
        evidence_count=int(row.get("evidence_count", 0)),
        aggregated_confidence=float(row.get("aggregated_confidence", 0)),
        aggregation_method=row.get("aggregation_method", ""),
        conflicting_evidence_count=int(row.get("conflicting_evidence_count", 0)),
        created_at=row.get("created_at", ""),
    )
