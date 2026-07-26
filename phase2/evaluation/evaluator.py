from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from phase2.evaluation.clusters import ClusterEvaluator
from phase2.evaluation.config import EvalConfig
from phase2.evaluation.document import DocumentEvaluator
from phase2.evaluation.embeddings import EmbeddingEvaluator
from phase2.evaluation.evidence import EvidenceEvaluator
from phase2.evaluation.observation import ObservationEvaluator
from phase2.evaluation.relationships import RelationshipEvaluator
from phase2.evaluation.schema import GlobalEvaluation, StageTiming
from phase2.evaluation.signals import SignalEvaluator


class EvaluationOrchestrator:
    def __init__(
        self,
        knowledge_dir: str | Path = "pain_intelligence/knowledge",
        config: EvalConfig | None = None,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._config = config or EvalConfig

        self._document = DocumentEvaluator(self._knowledge_dir)
        self._observation = ObservationEvaluator(self._knowledge_dir)
        self._evidence = EvidenceEvaluator(self._knowledge_dir)
        self._signals = SignalEvaluator(self._knowledge_dir)
        self._embeddings = EmbeddingEvaluator(self._knowledge_dir)
        self._relationships = RelationshipEvaluator(self._knowledge_dir)
        self._clusters = ClusterEvaluator(self._knowledge_dir)

    def evaluate(self) -> GlobalEvaluation:
        result = GlobalEvaluation()
        result.generated_at = datetime.now(timezone.utc).isoformat()

        timings: list[tuple[str, float, Any]] = []

        def _time_eval(name: str, fn):
            t0 = time.time()
            ev = fn
            elapsed = time.time() - t0
            timings.append((name, elapsed, ev))
            return ev

        # Phase 0: Documents
        doc_df = self._read_processed()
        result.documents = _time_eval("documents", self._document.evaluate(doc_df))

        # Phase 1.5: Observations, evidence, signals
        obs_df = self._read_asset("observations")
        doc_count = result.documents.total_documents
        result.observations = _time_eval("observations", self._observation.evaluate(obs_df, doc_count))

        ev_df = self._read_asset("evidence")
        result.evidence = _time_eval("evidence", self._evidence.evaluate(ev_df, result.observations.total_observations))

        sig_df = self._read_asset("problem_signals")
        result.signals = _time_eval("signals", self._signals.evaluate(sig_df))

        # Phase 2: Embeddings, relationships, clusters
        result.embeddings = _time_eval("embeddings", self._embeddings.evaluate())
        result.relationships = _time_eval("relationships", self._relationships.evaluate())
        result.clusters = _time_eval("clusters", self._clusters.evaluate())

        # Timing
        result.pipeline_timing = [
            StageTiming(stage=name, elapsed_seconds=round(el, 4))
            for name, el, _ in timings
        ]

        # Health score computation
        result = self._compute_overall_health(result)

        # Warnings & recommendations
        result.all_warnings = self._collect_warnings(result)
        result.recommendations = self._generate_recommendations(result)

        return result

    def _read_processed(self) -> pl.DataFrame:
        candidates = [
            self._knowledge_dir / "processed" / "processed.parquet",
            Path("outputs") / "processed.parquet",
        ]
        for c in candidates:
            if c.exists():
                return pl.read_parquet(str(c))
        return pl.DataFrame()

    def _read_asset(self, name: str) -> pl.DataFrame:
        path = self._knowledge_dir / "assets" / f"{name}.parquet"
        if path.exists():
            return pl.read_parquet(str(path))
        return pl.DataFrame()

    def _compute_overall_health(self, result: GlobalEvaluation) -> GlobalEvaluation:
        stages = [
            ("documents", result.documents.health),
            ("observations", result.observations.health),
            ("evidence", result.evidence.health),
            ("signals", result.signals.health),
            ("embeddings", result.embeddings.health),
            ("relationships", result.relationships.health),
            ("clusters", result.clusters.health),
        ]

        total_weight = 0.0
        weighted_sum = 0.0
        worst_stage = ""
        worst_score = 101.0

        for name, health in stages:
            w = health.weight
            total_weight += w
            weighted_sum += health.score * w
            if health.score < worst_score:
                worst_score = health.score
                worst_stage = name

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        result.overall_health_score = round(overall, 1)
        result.worst_stage = worst_stage

        # Compute degradation per stage
        for name, health in stages:
            degradation = 100.0 - health.score
            health.degradation_pct = round(degradation, 2)

        return result

    def _collect_warnings(self, result: GlobalEvaluation) -> list[str]:
        warnings: list[str] = []
        for stage_name in ["documents", "observations", "evidence", "signals", "embeddings", "relationships", "clusters"]:
            stage = getattr(result, stage_name)
            if hasattr(stage, "health") and stage.health.warnings:
                for w in stage.health.warnings:
                    warnings.append(f"[{stage_name}] {w}")
        return warnings

    def _generate_recommendations(self, result: GlobalEvaluation) -> list[str]:
        recs: list[str] = []

        if result.worst_stage and result.overall_health_score < 80:
            recs.append(f"Focus improvement on '{result.worst_stage}' — lowest health score")

        for stage_name in ["documents", "observations", "evidence", "signals", "embeddings", "relationships", "clusters"]:
            stage = getattr(result, stage_name)
            health = stage.health.score if hasattr(stage, "health") else 0
            if health < 50:
                recs.append(f"Critical attention needed: {stage_name} (score: {health})")

        if result.signals.accepted_signals == 0 and result.signals.zero_signal_explanation:
            recs.append(f"Investigate signal discovery: {result.signals.zero_signal_explanation}")

        if not recs and result.overall_health_score >= 80:
            recs.append("Pipeline quality is good. No immediate action required.")

        return recs
