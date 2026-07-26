"""Phase 1.5: Knowledge Extraction Engine orchestrator.

Pipeline: Observation Extraction → Knowledge Enrichment → Evidence → Problem Signals.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pain_intelligence.intelligence.config import get_intelligence_setting, load_intelligence_config
from pain_intelligence.intelligence.confidence import ConfidencePolicy
from pain_intelligence.intelligence.evidence import EvidenceBuilder, RuleAggregation
from pain_intelligence.intelligence.exporter import KnowledgeExporter
from pain_intelligence.intelligence.extractors.entities import EntityExtractor
from pain_intelligence.intelligence.extractors.keywords import KeywordExtractor
from pain_intelligence.intelligence.extractors.ngrams import NgramExtractor
from pain_intelligence.intelligence.extractors.patterns import PatternMatcher
from pain_intelligence.intelligence.extractors.phrases import PhraseExtractor
from pain_intelligence.intelligence.knowledge import KnowledgeEnricher
from pain_intelligence.intelligence.problem_signals import ProblemSignalDiscoverer
from pain_intelligence.intelligence.schema import Observation
from pain_intelligence.knowledge.manifest import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    PipelineManifest,
    compute_checksum,
    generate_run_id,
)
from pain_intelligence.knowledge.store import KnowledgeStore


class IntelligenceEngine:
    """Phase 1.5 orchestrator.

    Pipeline:
    1. Extract observations from all extractors
    2. Enrich observations with knowledge
    3. Build evidence from enriched observations
    4. Discover problem signals from evidence
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        debug: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.config = load_intelligence_config(config_path)
        self.debug = debug or get_intelligence_setting(self.config, "intelligence", "debug", default=False)

        # Run ID
        self._run_id = run_id or generate_run_id()

        # Knowledge store
        knowledge_dir = Path("pain_intelligence/knowledge")
        self.store = KnowledgeStore(knowledge_dir)
        self.store.run_id = self._run_id

        # Confidence policy
        conf_cfg = get_intelligence_setting(self.config, "intelligence", "confidence", default={})
        self.confidence = ConfidencePolicy(conf_cfg)

        # Extractors
        ext_cfg = get_intelligence_setting(self.config, "intelligence", "extraction", default={})
        self.extractors = [
            EntityExtractor(
                seed_entities=self.store.load_entities(),
                confidence=self.confidence,
                min_mentions=ext_cfg.get("entity_min_mentions", 3),
                pipeline_version=PIPELINE_VERSION,
            ),
            NgramExtractor(
                min_frequency=ext_cfg.get("ngram_min_frequency", 5),
                max_features=ext_cfg.get("ngram_max_features", 100),
                confidence=self.confidence,
                pipeline_version=PIPELINE_VERSION,
            ),
            KeywordExtractor(
                max_features=ext_cfg.get("keyword_max_features", 100),
                confidence=self.confidence,
                pipeline_version=PIPELINE_VERSION,
            ),
            PhraseExtractor(
                max_features=ext_cfg.get("ngram_max_features", 100),
                confidence=self.confidence,
                pipeline_version=PIPELINE_VERSION,
            ),
            PatternMatcher(
                patterns=self.store.load_patterns(),
                confidence=self.confidence,
                min_confidence=ext_cfg.get("pattern_min_confidence", 0.5),
                pipeline_version=PIPELINE_VERSION,
            ),
        ]

        # Reasoning modules
        self.enricher = KnowledgeEnricher(
            self.store,
            confidence=self.confidence,
            debug=self.debug,
        )
        self.evidence_builder = EvidenceBuilder(
            strategy=RuleAggregation(confidence=self.confidence)
        )

        # Configure adaptive thresholds
        ev_cfg = get_intelligence_setting(self.config, "intelligence", "evidence", default={})
        sig_cfg = get_intelligence_setting(self.config, "intelligence", "signals", default={})

        # These may be overridden dynamically based on dataset size
        self.signal_discoverer = ProblemSignalDiscoverer(
            store=self.store,
            min_document_count=sig_cfg.get("min_document_count"),
            max_avg_rating=sig_cfg.get("max_avg_rating", 3.0),
            min_confidence=sig_cfg.get("min_confidence", 0.7),
            confidence=self.confidence,
            pipeline_version=PIPELINE_VERSION,
        )

        # Exporter
        output_dir = get_intelligence_setting(self.config, "intelligence", "output_dir", default="reports")
        self.exporter = KnowledgeExporter(self.store, output_dir=output_dir)

        # Manifest (for the knowledge pipeline portion)
        self._manifest = PipelineManifest(knowledge_dir)

    @property
    def run_id(self) -> str:
        return self._run_id

    def run(self, data_path: str | None = None) -> dict[str, Any]:
        """Run the full knowledge extraction pipeline."""
        start = time.time()
        logger.info("Intelligence Engine v{} starting (run_id={}, debug={})", PIPELINE_VERSION, self._run_id, self.debug)

        df, input_checksum, doc_count = self._load_dataset(data_path)
        logger.info("Loaded {} documents (checksum={})", doc_count, input_checksum)

        # Set input metadata on exporter
        self.exporter.input_checksum = input_checksum
        self.exporter.input_document_count = doc_count

        # ── Stage 1: Observation Extraction ──
        logger.info("Stage 1: Extracting observations...")
        all_observations: list[Observation] = []
        for extractor in self.extractors:
            t0 = time.time()
            bundle = extractor.extract(df)
            elapsed = time.time() - t0
            all_observations.extend(bundle.observations)
            logger.info("  {} extracted {} observations in {:.2f}s",
                        extractor.name, len(bundle), elapsed)

        logger.info("Total observations: {}", len(all_observations))

        # ── Stage 2: Knowledge Enrichment ──
        logger.info("Stage 2: Enriching observations with knowledge...")
        t0 = time.time()
        resolution_log = self.enricher.enrich_batch(all_observations)
        logger.info("  Enriched {} observations in {:.2f}s (matched: {})",
                    len(all_observations), time.time() - t0,
                    sum(1 for r in resolution_log if r.matched))

        if self.debug:
            import json
            debug_dir = Path("reports")
            debug_dir.mkdir(exist_ok=True)
            debug_path = debug_dir / "resolution_debug.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in resolution_log], f, indent=2, default=str)
            logger.info("Wrote resolution debug log to {}", debug_path)

        # ── Stage 3: Evidence Construction ──
        logger.info("Stage 3: Building evidence...")
        t0 = time.time()

        # Apply adaptive thresholds to evidence builder
        adaptive_min_obs = max(3, int(doc_count ** 0.15))
        if hasattr(self.evidence_builder._strategy, '_min_observation_count'):
            self.evidence_builder._strategy._min_observation_count = adaptive_min_obs

        evidence = self.evidence_builder.build(all_observations)
        logger.info("  Built {} evidence records in {:.2f}s (adaptive min_obs={})",
                    len(evidence), time.time() - t0, adaptive_min_obs)

        # ── Stage 4: Problem Signal Discovery ──
        logger.info("Stage 4: Discovering problem signals...")
        t0 = time.time()

        # Apply adaptive thresholds to signal discoverer
        self.signal_discoverer.apply_adaptive_thresholds(doc_count)

        signals = self.signal_discoverer.discover(evidence)
        filtering_stats = self.signal_discoverer.filtering_stats
        logger.info("  Discovered {} problem signals in {:.2f}s", len(signals), time.time() - t0)
        if filtering_stats:
            logger.info("  Filtering: {} entity-only removed, {} generic removed",
                        filtering_stats.get("signals_removed_entity_only", 0),
                        filtering_stats.get("signals_removed_generic", 0))

        # ── Update manifest (before export, so run_id is shared) ──
        self._manifest.start_run(
            dataset_path=data_path or "outputs/processed.parquet",
        )
        self._run_id = self._manifest.run_id
        self.store.run_id = self._run_id

        # ── Export ──
        logger.info("Exporting assets...")
        exported = self.exporter.export(
            observations=all_observations,
            evidence=evidence,
            signals=signals,
            filtering_stats=filtering_stats,
        )
        for name, path in exported.items():
            logger.info("  Exported {}", name)

        # ── Write problem signal diagnostics ──
        self._write_signal_diagnostics(signals, filtering_stats, doc_count)
        for asset_name, asset_path in exported.items():
            ext = Path(asset_path).suffix
            if ext == ".parquet":
                import polars as pl2
                try:
                    rc = len(pl2.read_parquet(str(asset_path)))
                except Exception:
                    rc = 0
                self._manifest.register_asset(
                    name=asset_name,
                    path=asset_path,
                    record_count=rc,
                    stage="intelligence",
                )
        self._manifest.complete_run()
        self._manifest.save()
        logger.info("Manifest saved to {}", self._manifest.path)

        elapsed = time.time() - start
        logger.info("Pipeline complete in {:.2f}s", elapsed)

        result: dict[str, Any] = {
            "status": "completed",
            "run_id": self._run_id,
            "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": round(elapsed, 2),
            "observations_count": len(all_observations),
            "evidence_count": len(evidence),
            "signal_count": len(signals),
            "input_checksum": input_checksum,
            "input_document_count": doc_count,
            "filtering": {
                "signals_removed_entity_only": filtering_stats.get("signals_removed_entity_only", 0),
                "signals_removed_generic": filtering_stats.get("signals_removed_generic", 0),
                "signals_removed_total": sum(
                    filtering_stats.get(k, 0) for k in [
                        "signals_removed_entity_only",
                        "signals_removed_generic",
                        "signals_removed_low_documents",
                        "signals_removed_high_rating",
                        "signals_removed_low_confidence",
                    ]
                ) if filtering_stats else 0,
            } if filtering_stats else {},
            "adaptive_thresholds": self._get_adaptive_thresholds(doc_count),
        }
        return result

    def _load_dataset(self, data_path: str | None = None) -> tuple[pl.DataFrame, str, int]:
        if data_path is None:
            data_path = "outputs/processed.parquet"
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        checksum = compute_checksum(path)

        config_sampling = get_intelligence_setting(
            self.config, "intelligence", "sampling", default={}
        )
        if config_sampling.get("enabled", False):
            max_docs = config_sampling.get("max_documents", 100000)
            strategy = config_sampling.get("strategy", "stratified")
            if strategy == "stratified":
                strata = config_sampling.get("strata_column", "platform")
                df = pl.read_parquet(path)
                if strata in df.columns:
                    df = df.group_by(strata).agg(pl.all().sample(
                        n=min(max_docs // df[strata].n_unique(), df.height),
                        seed=42,
                    )).explode(pl.all().exclude(strata))
                else:
                    df = df.sample(n=min(max_docs, df.height), seed=42)
            else:
                df = pl.read_parquet(path).sample(n=min(max_docs, pl.read_parquet(path).height), seed=42)
            logger.info("Sampled {} documents (strategy={}, max={})", df.height, strategy, max_docs)
        else:
            df = pl.read_parquet(path)

        return df, checksum, df.height

    def _write_signal_diagnostics(
        self,
        signals: list,
        filtering_stats: dict[str, Any],
        doc_count: int,
    ) -> None:
        """Write problem signal diagnostics report."""
        import json

        diagnostics_dir = Path("reports")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Collect diagnostic information from signal discoverer
        diag = {
            "run_id": self._run_id,
            "pipeline_version": PIPELINE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "document_count": doc_count,
            "adaptive_thresholds": self._get_adaptive_thresholds(doc_count),
            "filtering": filtering_stats or {},
            "signals": [
                {
                    "signal_key": s.signal_key if hasattr(s, 'signal_key') else str(s),
                    "signal_text": s.signal_text if hasattr(s, 'signal_text') else "",
                    "category": s.category if hasattr(s, 'category') else "",
                    "entity": s.entity if hasattr(s, 'entity') else "",
                    "document_count": s.document_count if hasattr(s, 'document_count') else 0,
                    "confidence": s.confidence if hasattr(s, 'confidence') else 0.0,
                    "observation_count": s.observation_count if hasattr(s, 'observation_count') else 0,
                }
                for s in signals
            ],
        }

        path = diagnostics_dir / "problem_signal_diagnostics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, default=str, ensure_ascii=False)
        logger.info("Wrote problem signal diagnostics to {}", path)

    @staticmethod
    def _get_adaptive_thresholds(doc_count: int) -> dict[str, float]:
        import math
        return {
            "support_threshold": max(3, int(math.log10(max(doc_count, 1)))),
            "min_observation_count": max(3, int(doc_count ** 0.15)),
            "min_confidence": 0.7,
            "max_avg_rating": 3.0,
            "evidence_min_group_size": 3,
        }

    def _get_threshold_info(self) -> dict[str, Any]:
        """Return current threshold configuration."""
        sig = self.signal_discoverer
        return {
            "min_document_count": sig.min_document_count,
            "max_avg_rating": sig.max_avg_rating,
            "min_confidence": sig.min_confidence,
            "entity_names_count": len(sig._entity_names),
            "generic_phrases_count": len(sig._generic_phrases),
        }
