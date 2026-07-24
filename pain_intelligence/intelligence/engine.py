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
    ) -> None:
        self.config = load_intelligence_config(config_path)
        self.debug = debug or get_intelligence_setting(self.config, "intelligence", "debug", default=False)
        self.pipeline_version = "1.5.0"

        # Knowledge store
        knowledge_dir = Path("pain_intelligence/knowledge")
        self.store = KnowledgeStore(knowledge_dir)
        self.store.pipeline_version = self.pipeline_version

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
                pipeline_version=self.pipeline_version,
            ),
            NgramExtractor(
                min_frequency=ext_cfg.get("ngram_min_frequency", 5),
                max_features=ext_cfg.get("ngram_max_features", 100),
                confidence=self.confidence,
                pipeline_version=self.pipeline_version,
            ),
            KeywordExtractor(
                max_features=ext_cfg.get("keyword_max_features", 100),
                confidence=self.confidence,
                pipeline_version=self.pipeline_version,
            ),
            PhraseExtractor(
                max_features=ext_cfg.get("ngram_max_features", 100),
                confidence=self.confidence,
                pipeline_version=self.pipeline_version,
            ),
            PatternMatcher(
                patterns=self.store.load_patterns(),
                confidence=self.confidence,
                min_confidence=ext_cfg.get("pattern_min_confidence", 0.5),
                pipeline_version=self.pipeline_version,
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

        ev_cfg = get_intelligence_setting(self.config, "intelligence", "evidence", default={})
        sig_cfg = get_intelligence_setting(self.config, "intelligence", "signals", default={})
        self.signal_discoverer = ProblemSignalDiscoverer(
            store=self.store,
            min_document_count=sig_cfg.get("min_document_count", 10),
            max_avg_rating=sig_cfg.get("max_avg_rating", 3.0),
            min_confidence=sig_cfg.get("min_confidence", 0.7),
            confidence=self.confidence,
            pipeline_version=self.pipeline_version,
        )

        # Exporter
        output_dir = get_intelligence_setting(self.config, "intelligence", "output_dir", default="reports")
        self.exporter = KnowledgeExporter(self.store, output_dir=output_dir)

    def run(self, data_path: str | None = None) -> dict[str, Any]:
        """Run the full knowledge extraction pipeline."""
        start = time.time()
        logger.info("Intelligence Engine v{} starting (debug={})", self.pipeline_version, self.debug)

        df = self._load_dataset(data_path)
        logger.info("Loaded {} documents", df.height)

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
            # Write debug resolution log
            import json
            from pathlib import Path
            debug_dir = Path("reports")
            debug_dir.mkdir(exist_ok=True)
            debug_path = debug_dir / "resolution_debug.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in resolution_log], f, indent=2, default=str)
            logger.info("Wrote resolution debug log to {}", debug_path)

        # ── Stage 3: Evidence Construction ──
        logger.info("Stage 3: Building evidence...")
        t0 = time.time()
        evidence = self.evidence_builder.build(all_observations)
        logger.info("  Built {} evidence records in {:.2f}s", len(evidence), time.time() - t0)

        # ── Stage 4: Problem Signal Discovery ──
        logger.info("Stage 4: Discovering problem signals...")
        t0 = time.time()
        signals = self.signal_discoverer.discover(evidence)
        filtering_stats = self.signal_discoverer.filtering_stats
        logger.info("  Discovered {} problem signals in {:.2f}s", len(signals), time.time() - t0)
        if filtering_stats:
            logger.info("  Filtering: {} entity-only removed, {} generic removed",
                        filtering_stats.get("signals_removed_entity_only", 0),
                        filtering_stats.get("signals_removed_generic", 0))

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

        elapsed = time.time() - start
        logger.info("Pipeline complete in {:.2f}s", elapsed)

        return {
            "status": "completed",
            "pipeline_version": self.pipeline_version,
            "elapsed_seconds": round(elapsed, 2),
            "observations_count": len(all_observations),
            "evidence_count": len(evidence),
            "signal_count": len(signals),
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
        }

    def _load_dataset(self, data_path: str | None = None) -> pl.DataFrame:
        if data_path is None:
            data_path = "outputs/processed.parquet"
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

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

        return df