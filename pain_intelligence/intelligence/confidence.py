"""Central confidence scoring policy for the Knowledge Extraction Engine.

Every module calls ConfidencePolicy instead of computing confidence inline.
Ensures consistent, configurable scoring across all extraction methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pain_intelligence.intelligence.schema import Evidence, ExtractionMethod


@dataclass
class ExtractionContext:
    """Context for computing extraction confidence."""
    method: ExtractionMethod = ExtractionMethod.HEURISTIC
    exact_match: bool = False
    frequency: int = 1
    source_diversity: int = 1
    rating_alignment: float = 0.5


@dataclass
class EvidenceStats:
    """Statistics for computing evidence confidence."""
    observation_count: int = 0
    document_count: int = 0
    rating_std: float = 0.0
    platform_diversity: int = 1
    country_diversity: int = 1
    avg_observation_confidence: float = 0.0


class ConfidencePolicy:
    """Central confidence scoring policy.
    
    Default scores can be overridden via config.
    All scores are 0.0 (low) to 1.0 (high).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._scores: dict[str, float] = {
            "exact_dictionary_match": 0.95,
            "pattern_match": 0.85,
            "fuzzy_match": 0.65,
            "heuristic": 0.55,
            "statistical": 0.45,
            "evidence_base": 0.70,
            "signal_threshold": 0.80,
        }
        self._scores.update(cfg.get("confidence", {}))

    # ── Extraction confidence ─────────────────────────────────

    def for_extraction(self, context: ExtractionContext) -> float:
        """Compute confidence for an extracted observation."""
        if context.method == ExtractionMethod.DICTIONARY_MATCH:
            base = self._scores["exact_dictionary_match"]
            if not context.exact_match:
                base *= 0.7  # fuzzy match penalty
            return self._apply_boost(base, context)

        if context.method == ExtractionMethod.PATTERN_MATCH:
            return self._scores["pattern_match"]

        if context.method == ExtractionMethod.HEURISTIC:
            base = self._scores["heuristic"]
            return self._apply_boost(base, context)

        if context.method == ExtractionMethod.STATISTICAL:
            return self._scores["statistical"]

        return 0.0

    def for_pattern_match(self, pattern_label: str, match_confidence: float) -> float:
        """Pattern base confidence adjusted by match quality."""
        base = self._scores["pattern_match"]
        return base * match_confidence

    # ── Evidence confidence ───────────────────────────────────

    def for_evidence(self, stats: EvidenceStats) -> float:
        """Compute evidence confidence from aggregated statistics."""
        if stats.document_count == 0:
            return 0.0

        base = self._scores["evidence_base"]

        count_factor = min(1.0, stats.document_count / 100.0)
        diversity_factor = min(1.0, (stats.platform_diversity + stats.country_diversity) / 8.0)
        rating_factor = 1.0 - min(1.0, stats.rating_std / 2.5)
        obs_conf_factor = min(1.0, stats.avg_observation_confidence)
        obs_count_factor = min(1.0, stats.observation_count / 500.0)

        confidence = (
            base * 0.25
            + count_factor * 0.20
            + diversity_factor * 0.15
            + rating_factor * 0.10
            + obs_conf_factor * 0.15
            + obs_count_factor * 0.15
        )
        return round(min(1.0, confidence), 4)

    # ── Signal confidence ─────────────────────────────────────

    def for_signal(self, evidence_list: list[Evidence]) -> float:
        """Compute problem signal confidence from supporting evidence.

        Factors:
          - Average evidence confidence (core)
          - Number of supporting evidence records (strength)
          - Total document count (reach)
          - Average rating (lower = stronger problem signal)
        """
        if not evidence_list:
            return 0.0

        avg_ev_conf = sum(e.confidence for e in evidence_list) / len(evidence_list)
        strength = min(1.0, len(evidence_list) / 5.0)

        total_docs = sum(e.document_count for e in evidence_list)
        doc_factor = min(1.0, total_docs / 500.0)

        ratings = [e.avg_rating for e in evidence_list if e.avg_rating is not None]
        rating_factor = 1.0
        if ratings:
            avg_r = sum(ratings) / len(ratings)
            # Lower rating → higher confidence (real problem)
            rating_factor = 1.0 - min(1.0, avg_r / 5.0)

        threshold = self._scores["signal_threshold"]
        confidence = (
            avg_ev_conf * 0.35
            + strength * 0.20
            + doc_factor * 0.25
            + rating_factor * 0.20
        )
        return round(min(1.0, confidence), 4)

    # ── Internal ──────────────────────────────────────────────

    def _apply_boost(self, base: float, ctx: ExtractionContext) -> float:
        """Apply frequency and diversity boosts."""
        freq_boost = min(0.1, ctx.frequency / 1000.0 * 0.05)
        div_boost = min(0.05, ctx.source_diversity / 100.0 * 0.03)
        return round(min(1.0, base + freq_boost + div_boost), 4)

    def __repr__(self) -> str:
        return f"ConfidencePolicy(scores={self._scores})"