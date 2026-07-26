"""Problem signal discovery from evidence.

Pipeline:
  Evidence → Merge semantic variants → Filter entity-only → Filter generic
    → Validate category → Score confidence → Build ProblemSignal
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from pain_intelligence.intelligence.confidence import ConfidencePolicy
from pain_intelligence.intelligence.schema import (
    EntityType,
    Evidence,
    ProblemSignal,
    signal_id,
)
from pain_intelligence.knowledge.store import KnowledgeStore


class ProblemSignalDiscoverer:
    """Discovers problem signals from aggregated evidence.

    Filters out:
      - Entity-only signals (no associated pain expression)
      - Generic sentiment-only phrases

    Validates:
      - Minimum document count (adaptive: max(3, log10(doc_count)))
      - Maximum average rating
      - Minimum confidence threshold
      - Category presence

    Merges semantic variants via canonical signal concept.
    """

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        min_document_count: int | None = None,
        max_avg_rating: float = 3.0,
        min_confidence: float = 0.7,
        confidence: ConfidencePolicy | None = None,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.max_avg_rating = max_avg_rating
        self.min_confidence = min_confidence
        self.confidence = confidence or ConfidencePolicy()
        self.pipeline_version = pipeline_version

        # Adaptive thresholds — store the configured value but override
        # dynamically via apply_adaptive_thresholds()
        self._configured_min_document_count = min_document_count
        self.min_document_count = min_document_count if min_document_count is not None else 10

        # Load filtering data from store if available
        self._entity_names: set[str] = set()
        self._generic_phrases: set[str] = set()
        if store:
            for e in store.load_entities():
                name = e.get("name", "").lower().strip()
                if name:
                    self._entity_names.add(name)
            self._generic_phrases = set(
                p.lower().strip() for p in store.load_generic_sentiment() if p.strip()
            )

        # Store last diagnostics
        self._last_filtering_stats: dict[str, Any] = {}
        self._last_discarded: list[dict[str, Any]] = []
        self._last_support_distribution: list[int] = []
        self._last_confidence_distribution: list[float] = []
        self._last_evidence_distribution: list[int] = []

    @property
    def filtering_stats(self) -> dict[str, Any]:
        return dict(self._last_filtering_stats)

    def apply_adaptive_thresholds(self, document_count: int) -> None:
        """Scale thresholds based on dataset size.

        Rules:
          - support_threshold = max(3, log10(document_count))
          - min_observation_count = max(3, document_count ** 0.15)
          - min_confidence stays at configured value
          - max_avg_rating stays at configured value
        """
        self.min_document_count = max(
            3,
            int(math.log10(max(document_count, 1))),
        )
        if self._configured_min_document_count is not None:
            self.min_document_count = max(
                self._configured_min_document_count,
                self.min_document_count,
            )
        # Store thresholds in filtering stats for immediate access
        self._last_filtering_stats["thresholds"] = {
            "min_document_count": self.min_document_count,
            "max_avg_rating": self.max_avg_rating,
            "min_confidence": self.min_confidence,
        }

    def discover(self, evidence_list: list[Evidence]) -> list[ProblemSignal]:
        """Filter and aggregate evidence into problem signals.

        Returns filtered, merged, and scored problem signals.
        """
        # Reset diagnostics
        self._last_discarded = []

        # Track filtering stats
        support_counts: list[int] = []
        confidence_scores: list[float] = []
        evidence_counts: list[int] = []

        stats: dict[str, Any] = {
            "total_evidence_input": len(evidence_list),
            "signals_removed_entity_only": 0,
            "signals_removed_generic": 0,
            "signals_removed_entity_only_details": [],
            "signals_removed_generic_details": [],
            "signals_removed_low_documents": 0,
            "signals_removed_low_documents_details": [],
            "signals_removed_high_rating": 0,
            "signals_removed_high_rating_details": [],
            "signals_removed_low_confidence": 0,
            "signals_removed_low_confidence_details": [],
            "signals_removed_no_category": 0,
            "signals_removed_no_category_details": [],
            "thresholds": {
                "min_document_count": self.min_document_count,
                "max_avg_rating": self.max_avg_rating,
                "min_confidence": self.min_confidence,
            },
        }

        # ── 1. Pre-filter by thresholds ──
        candidates = []
        for ev in evidence_list:
            if ev.document_count < self.min_document_count:
                stats["signals_removed_low_documents"] += 1
                stats["signals_removed_low_documents_details"].append({
                    "signal": ev.signal_text,
                    "entity": ev.entity,
                    "support": ev.document_count,
                    "confidence": ev.confidence,
                    "reason": "below minimum document count",
                    "rule": f"document_count < {self.min_document_count}",
                    "score": float(ev.document_count),
                })
                continue

            if ev.avg_rating is not None and ev.avg_rating > self.max_avg_rating:
                stats["signals_removed_high_rating"] += 1
                stats["signals_removed_high_rating_details"].append({
                    "signal": ev.signal_text,
                    "entity": ev.entity,
                    "support": ev.document_count,
                    "confidence": ev.confidence,
                    "reason": "above max average rating",
                    "rule": f"avg_rating > {self.max_avg_rating}",
                    "score": float(ev.avg_rating),
                })
                continue

            # ── 2. Filter entity-only signals ──
            if self._is_entity_only(ev):
                stats["signals_removed_entity_only"] += 1
                stats["signals_removed_entity_only_details"].append({
                    "signal": ev.signal_text,
                    "entity": ev.entity,
                    "support": ev.document_count,
                    "confidence": ev.confidence,
                    "reason": "entity-only signal (no pain expression)",
                    "rule": "signal_text is entity name",
                    "score": float(ev.confidence),
                })
                continue

            # ── 3. Filter generic sentiment-only phrases ──
            if self._is_generic_phrase(ev):
                stats["signals_removed_generic"] += 1
                stats["signals_removed_generic_details"].append({
                    "signal": ev.signal_text,
                    "entity": ev.entity,
                    "support": ev.document_count,
                    "confidence": ev.confidence,
                    "reason": "generic sentiment-only phrase",
                    "rule": "signal_text in generic_phrases set",
                    "score": float(ev.confidence),
                })
                continue

            # ── 4. Validate category presence ──
            if not ev.category:
                stats["signals_removed_no_category"] += 1
                stats["signals_removed_no_category_details"].append({
                    "signal": ev.signal_text,
                    "entity": ev.entity,
                    "support": ev.document_count,
                    "confidence": ev.confidence,
                    "reason": "no category assigned",
                    "rule": "category is None",
                    "score": float(ev.confidence),
                })
                continue

            candidates.append(ev)

            # Collect distribution data
            support_counts.append(ev.document_count)
            confidence_scores.append(ev.confidence)
            evidence_counts.append(ev.observation_count)

        # ── 5. Merge semantic variants (group by canonical signal concept) ──
        merged: dict[str, list[Evidence]] = {}
        for ev in candidates:
            canonical_key = self._canonical_key(ev)
            if canonical_key not in merged:
                merged[canonical_key] = []
            merged[canonical_key].append(ev)

        # ── 6. Build ProblemSignal for each merged group ──
        signals: list[ProblemSignal] = []
        for canonical_key, ev_group in merged.items():
            total_docs = sum(e.document_count for e in ev_group)
            total_obs = sum(e.observation_count for e in ev_group)
            avg_rating = self._weighted_avg_rating(ev_group)
            all_ev_ids = []
            for e in ev_group:
                all_ev_ids.extend(e.evidence_id for _ in range(e.observation_count))

            entity_counter = Counter(e.entity for e in ev_group if e.entity)
            best_entity = entity_counter.most_common(1)[0][0] if entity_counter else None

            cat_counter = Counter(e.category for e in ev_group if e.category)
            best_category = cat_counter.most_common(1)[0][0] if cat_counter else None

            signal_text = self._derive_signal_text(canonical_key, ev_group)

            signal_confidence = self.confidence.for_signal(ev_group)

            if signal_confidence < self.min_confidence:
                stats["signals_removed_low_confidence"] += 1
                stats["signals_removed_low_confidence_details"].append({
                    "signal": signal_text,
                    "entity": best_entity,
                    "support": total_docs,
                    "confidence": signal_confidence,
                    "reason": "below minimum confidence threshold",
                    "rule": f"confidence < {self.min_confidence}",
                    "score": float(signal_confidence),
                })
                continue

            signals.append(ProblemSignal(
                signal_key=signal_id(
                    canonical_key,
                    best_entity or "",
                    "",
                ),
                category=best_category,
                entity=best_entity,
                entity_type=EntityType.UNKNOWN,
                signal_text=signal_text,
                document_count=total_docs,
                avg_rating=avg_rating,
                evidence_ids=list(set(all_ev_ids)),
                observation_count=total_obs,
                confidence=signal_confidence,
                pipeline_version=self.pipeline_version,
                generated_at=datetime.now(timezone.utc).isoformat(),
            ))

        result = sorted(signals, key=lambda x: -x.confidence)
        self._last_filtering_stats = stats
        self._last_support_distribution = sorted(support_counts)
        self._last_confidence_distribution = sorted(confidence_scores)
        self._last_evidence_distribution = sorted(evidence_counts)

        # Build full discard log
        self._last_discarded = (
            stats.get("signals_removed_low_documents_details", []) +
            stats.get("signals_removed_high_rating_details", []) +
            stats.get("signals_removed_entity_only_details", []) +
            stats.get("signals_removed_generic_details", []) +
            stats.get("signals_removed_no_category_details", []) +
            stats.get("signals_removed_low_confidence_details", [])
        )

        return result

    def get_diagnostics(self) -> dict[str, Any]:
        """Return full diagnostics from the last discover() run.

        Includes distribution data and per-discarded-signal explainability.
        """
        stats = dict(self._last_filtering_stats)
        total_removed = sum(
            stats.get(k, 0) for k in [
                "signals_removed_entity_only",
                "signals_removed_generic",
                "signals_removed_low_documents",
                "signals_removed_high_rating",
                "signals_removed_low_confidence",
                "signals_removed_no_category",
            ]
        )
        return {
            "candidate_count_before_filtering": stats.get("total_evidence_input", 0),
            "total_removed": total_removed,
            "remaining_count": stats.get("total_evidence_input", 0) - total_removed,
            "removal_by_reason": {
                "entity_only": stats.get("signals_removed_entity_only", 0),
                "generic_phrase": stats.get("signals_removed_generic", 0),
                "low_document_count": stats.get("signals_removed_low_documents", 0),
                "high_rating": stats.get("signals_removed_high_rating", 0),
                "low_confidence": stats.get("signals_removed_low_confidence", 0),
                "no_category": stats.get("signals_removed_no_category", 0),
            },
            "thresholds": stats.get("thresholds", {}),
            "support_distribution": {
                "min": min(self._last_support_distribution) if self._last_support_distribution else 0,
                "max": max(self._last_support_distribution) if self._last_support_distribution else 0,
                "count": len(self._last_support_distribution),
                "values": self._last_support_distribution[:100],
            },
            "confidence_distribution": {
                "min": min(self._last_confidence_distribution) if self._last_confidence_distribution else 0,
                "max": max(self._last_confidence_distribution) if self._last_confidence_distribution else 0,
                "count": len(self._last_confidence_distribution),
                "values": [round(v, 4) for v in self._last_confidence_distribution[:100]],
            },
            "evidence_distribution": {
                "min": min(self._last_evidence_distribution) if self._last_evidence_distribution else 0,
                "max": max(self._last_evidence_distribution) if self._last_evidence_distribution else 0,
                "count": len(self._last_evidence_distribution),
                "values": self._last_evidence_distribution[:100],
            },
            "discarded_signals": self._last_discarded[:200],
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _is_entity_only(self, ev: Evidence) -> bool:
        signal_text = (ev.signal_text or "").lower().strip()
        entity = (ev.entity or "").lower().strip()

        if signal_text and entity and signal_text == entity:
            return True
        if signal_text in self._entity_names:
            return True
        return False

    def _is_generic_phrase(self, ev: Evidence) -> bool:
        signal_text = (ev.signal_text or "").lower().strip()
        if signal_text in self._generic_phrases:
            return True
        return False

    @staticmethod
    def _canonical_key(ev: Evidence) -> str:
        entity = (ev.entity or "").lower().strip()
        text = (ev.signal_text or "").lower().strip()
        if entity:
            return f"{text}:{entity}"
        return text

    @staticmethod
    def _derive_signal_text(canonical_key: str, ev_group: list[Evidence]) -> str:
        text_counts: Counter[str] = Counter()
        for ev in ev_group:
            txt = (ev.signal_text or "").strip()
            if txt:
                text_counts[txt] += ev.observation_count
        if text_counts:
            return text_counts.most_common(1)[0][0]
        parts = canonical_key.split(":")
        return parts[0] if parts else canonical_key

    @staticmethod
    def _weighted_avg_rating(ev_group: list[Evidence]) -> float | None:
        ratings = []
        for ev in ev_group:
            if ev.avg_rating is not None:
                ratings.extend([ev.avg_rating] * ev.observation_count)
        if not ratings:
            return None
        return sum(ratings) / len(ratings)
