"""Problem signal discovery from evidence.

Pipeline:
  Evidence → Merge semantic variants → Filter entity-only → Filter generic
    → Validate category → Score confidence → Build ProblemSignal
"""

from __future__ import annotations

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
      - Minimum document count
      - Maximum average rating
      - Minimum confidence threshold
      - Category presence

    Merges semantic variants via canonical signal concept.
    """

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        min_document_count: int = 10,
        max_avg_rating: float = 3.0,
        min_confidence: float = 0.7,
        confidence: ConfidencePolicy | None = None,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.min_document_count = min_document_count
        self.max_avg_rating = max_avg_rating
        self.min_confidence = min_confidence
        self.confidence = confidence or ConfidencePolicy()
        self.pipeline_version = pipeline_version

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

    @property
    def filtering_stats(self) -> dict[str, Any]:
        """Return filtering statistics from the last discover() run."""
        return dict(self._last_filtering_stats)

    def discover(self, evidence_list: list[Evidence]) -> list[ProblemSignal]:
        """Filter and aggregate evidence into problem signals.

        Returns filtered, merged, and scored problem signals.
        """
        # Track filtering stats
        stats: dict[str, Any] = {
            "total_evidence_input": len(evidence_list),
            "signals_removed_entity_only": 0,
            "signals_removed_generic": 0,
            "signals_removed_entity_only_details": [],
            "signals_removed_generic_details": [],
            "signals_removed_low_documents": 0,
            "signals_removed_high_rating": 0,
            "signals_removed_low_confidence": 0,
        }

        # ── 1. Pre-filter by thresholds ──
        candidates = []
        for ev in evidence_list:
            if ev.document_count < self.min_document_count:
                stats["signals_removed_low_documents"] += 1
                continue
            if ev.avg_rating is not None and ev.avg_rating > self.max_avg_rating:
                stats["signals_removed_high_rating"] += 1
                continue

            # ── 2. Filter entity-only signals ──
            if self._is_entity_only(ev):
                stats["signals_removed_entity_only"] += 1
                stats["signals_removed_entity_only_details"].append({
                    "signal_text": ev.signal_text,
                    "entity": ev.entity,
                    "document_count": ev.document_count,
                })
                continue

            # ── 3. Filter generic sentiment-only phrases ──
            if self._is_generic_phrase(ev):
                stats["signals_removed_generic"] += 1
                stats["signals_removed_generic_details"].append({
                    "signal_text": ev.signal_text,
                    "entity": ev.entity,
                    "document_count": ev.document_count,
                })
                continue

            candidates.append(ev)

        # ── 4. Merge semantic variants (group by canonical signal concept) ──
        merged: dict[str, list[Evidence]] = {}
        for ev in candidates:
            canonical_key = self._canonical_key(ev)
            if canonical_key not in merged:
                merged[canonical_key] = []
            merged[canonical_key].append(ev)

        # ── 5. Build ProblemSignal for each merged group ──
        signals: list[ProblemSignal] = []
        for canonical_key, ev_group in merged.items():
            # Merge evidence into one signal
            total_docs = sum(e.document_count for e in ev_group)
            total_obs = sum(e.observation_count for e in ev_group)
            avg_rating = self._weighted_avg_rating(ev_group)
            all_ev_ids = []
            for e in ev_group:
                all_ev_ids.extend(e.evidence_id for _ in range(e.observation_count))

            # Pick best entity (most frequent across evidence)
            entity_counter = Counter(e.entity for e in ev_group if e.entity)
            best_entity = entity_counter.most_common(1)[0][0] if entity_counter else None

            # Pick best category (most frequent across evidence)
            cat_counter = Counter(e.category for e in ev_group if e.category)
            best_category = cat_counter.most_common(1)[0][0] if cat_counter else None

            # Derive signal text from canonical_key or best evidence
            signal_text = self._derive_signal_text(canonical_key, ev_group)

            # Confidence: combine evidence scores weighted by document count
            signal_confidence = self.confidence.for_signal(ev_group)

            # Final confidence threshold
            if signal_confidence < self.min_confidence:
                stats["signals_removed_low_confidence"] += 1
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
        return result

    # ── Internal helpers ──────────────────────────────────────────

    def _is_entity_only(self, ev: Evidence) -> bool:
        """Check if evidence is a standalone entity with no pain expression."""
        signal_text = (ev.signal_text or "").lower().strip()
        entity = (ev.entity or "").lower().strip()

        # If the signal text is identical to the entity name, it's entity-only
        if signal_text and entity and signal_text == entity:
            return True

        # If the signal text is a known entity name
        if signal_text in self._entity_names:
            return True

        return False

    def _is_generic_phrase(self, ev: Evidence) -> bool:
        """Check if evidence is a generic sentiment-only phrase."""
        signal_text = (ev.signal_text or "").lower().strip()
        if signal_text in self._generic_phrases:
            return True
        return False

    @staticmethod
    def _canonical_key(ev: Evidence) -> str:
        """Derive a canonical grouping key from evidence.

        Uses signal_text (which is already canonical from KnowledgeEnricher)
        + entity for grouping.
        """
        entity = (ev.entity or "").lower().strip()
        text = (ev.signal_text or "").lower().strip()
        if entity:
            return f"{text}:{entity}"
        return text

    @staticmethod
    def _derive_signal_text(canonical_key: str, ev_group: list[Evidence]) -> str:
        """Derive the best signal text for a merged group."""
        # Use the most common signal_text across evidence records
        text_counts: Counter[str] = Counter()
        for ev in ev_group:
            txt = (ev.signal_text or "").strip()
            if txt:
                text_counts[txt] += ev.observation_count
        if text_counts:
            return text_counts.most_common(1)[0][0]
        # Fallback to canonical key (without entity suffix)
        parts = canonical_key.split(":")
        return parts[0] if parts else canonical_key

    @staticmethod
    def _weighted_avg_rating(ev_group: list[Evidence]) -> float | None:
        """Compute weighted average rating across evidence."""
        ratings = []
        for ev in ev_group:
            if ev.avg_rating is not None:
                ratings.extend([ev.avg_rating] * ev.observation_count)
        if not ratings:
            return None
        return sum(ratings) / len(ratings)