"""Knowledge enrichment: resolves raw observations with seed + discovered knowledge.

Converts observations into enriched records with:
  - entity resolution (known entity names, aliases)
  - category assignment (pattern labels, taxonomy keywords, problem signal concepts)
  - canonicalization (mapping to canonical problem signal concepts)
  - alias resolution (alias → canonical entity)
"""

from __future__ import annotations

import time
from typing import Any

from pain_intelligence.intelligence.confidence import ConfidencePolicy
from pain_intelligence.intelligence.schema import (
    DebugResolutionResult,
    EntityType,
    Observation,
    ResolutionResult,
)
from pain_intelligence.knowledge.store import KnowledgeStore


class KnowledgeEnricher:
    """Enriches raw observations with entity, category, type, and canonical signal knowledge.

    Responsibilities:
      - Entity resolution (seed dictionary + aliases)
      - Category assignment (pattern labels, taxonomy keywords, canonical concepts)
      - Canonicalization (mapping phrases to canonical signal concepts)
      - Alias resolution

    Mutates observations in-place. Returns ResolutionResult metadata.
    In debug mode, returns DebugResolutionResult with full candidate info.
    """

    # Mapping from pattern label → (canonical_value, category)
    # Lives here (not in patterns.yaml) so patterns remain pure detection rules.
    _PATTERN_LABEL_MAP: dict[str, tuple[str, str | None]] = {
        "delivery_delay": ("Late Delivery", "Delivery"),
        "refund_issue": ("Refund Not Received", "Refund"),
        "defective_product": ("Defective Product", "Quality"),
        "customer_support": ("Customer Support Not Responding", "Customer Support"),
        "account_issue": ("Account Issue", "Account"),
        "payment_failed": ("Payment Failed", "Payment"),
        "overcharged": ("Overcharged", "Pricing"),
        "poor_quality": ("Poor Quality", "Quality"),
        "missing_feature": ("Missing Feature", "Feature"),
        "improvement_request": ("Improvement Needed", "Feature"),
        "switched_from": ("Switched From Competitor", None),
        "better_than": ("Better Than Competitor", None),
        "alternative": ("Considered Alternative", None),
        "urgent": ("Urgent Issue", None),
    }

    def __init__(
        self,
        store: KnowledgeStore,
        confidence: ConfidencePolicy | None = None,
        debug: bool = False,
    ) -> None:
        self.store = store
        self.confidence = confidence or ConfidencePolicy()
        self.debug = debug
        self._entities: list[dict[str, Any]] = store.load_entities()
        self._patterns: list[dict[str, Any]] = store.load_patterns()
        self._taxonomy: dict[str, dict[str, Any]] = store.load_taxonomy()
        self._problem_signals: dict[str, dict[str, Any]] = store.load_problem_signals()
        self._generic_sentiment: list[str] = store.load_generic_sentiment()
        self._build_lookups()

    def _build_lookups(self) -> None:
        self._entity_names: set[str] = {
            e["name"].lower() for e in self._entities
        }
        self._entity_aliases: dict[str, str] = {}
        for e in self._entities:
            name = e["name"].lower()
            for alias in e.get("aliases", []):
                self._entity_aliases[alias.lower()] = name
        self._taxonomy_keywords: dict[str, list[str]] = {}
        for cat, data in self._taxonomy.items():
            self._taxonomy_keywords[cat] = [k.lower() for k in data.get("keywords", [])]

        # Build canonical concept lookup: category name → canonical concept name
        # from problem_signals.yaml
        self._canonical_by_category: dict[str, str] = {}
        for concept, info in self._problem_signals.items():
            category = info.get("category", "")
            if category:
                self._canonical_by_category[category] = concept

    def enrich(self, obs: Observation) -> ResolutionResult:
        """Enrich a single observation in-place. Returns resolution metadata."""
        start = time.time()
        result = ResolutionResult(
            observation_id=obs.observation_id,
            original_value=obs.value,
            matched=False,
            method="unresolved",
            confidence=0.0,
        )

        value_lower = obs.value.lower()

        # ── 1. Entity matching ──
        if value_lower in self._entity_names:
            obs.entity = value_lower
            obs.entity_type = EntityType.COMPANY
            result.matched = True
            result.method = "seed_dictionary"
            result.confidence = self.confidence.for_extraction(
                type("ctx", (), {"method": type("m", (), {"value": "dictionary_match"})(), "exact_match": True})()
            )
            result.resolved_entity = value_lower
            result.resolved_type = EntityType.COMPANY

        elif value_lower in self._entity_aliases:
            canonical = self._entity_aliases[value_lower]
            obs.entity = canonical
            obs.entity_type = EntityType.COMPANY
            result.matched = True
            result.method = "alias_match"
            result.confidence = self.confidence.for_extraction(
                type("ctx", (), {"method": type("m", (), {"value": "dictionary_match"})(), "exact_match": False})()
            )
            result.resolved_entity = canonical
            result.resolved_type = EntityType.COMPANY

        # ── 2. Pattern label → category + canonical_value ──
        if obs.pattern_label:
            label_info = self._PATTERN_LABEL_MAP.get(obs.pattern_label)
            if label_info:
                canonical_val, category_val = label_info
                obs.canonical_value = canonical_val
                obs.canonical_source = "pattern"
                if category_val:
                    obs.category = category_val
                    result.resolved_category = category_val
                result.matched = True
                result.method = f"pattern_label:{obs.pattern_label}"
                result.confidence = max(result.confidence, 0.85)

        # ── 3. Category assignment via taxonomy keywords (if not already set) ──
        if not obs.category:
            for cat, keywords in self._taxonomy_keywords.items():
                for kw in keywords:
                    # Word-boundary-aware matching to avoid false positives
                    if self._word_boundary_match(kw, value_lower):
                        obs.category = cat
                        result.resolved_category = cat
                        if not result.matched:
                            result.matched = True
                            result.method = "taxonomy_match"
                            result.confidence = 0.7
                        break
                if obs.category:
                    break

        # ── 4. Canonicalization (if not already set) ──
        if not obs.canonical_value:
            # Try canonical concept lookup by category
            if obs.category and obs.category in self._canonical_by_category:
                # Use the canonical concept name from problem_signals
                concept = self._canonical_by_category[obs.category]
                obs.canonical_value = concept
                obs.canonical_source = "problem_signals"
            elif obs.category:
                # Fallback: use category name as signal
                obs.canonical_value = obs.category
                obs.canonical_source = "taxonomy"
            elif obs.pattern_label:
                # Pattern label observed but not in our map; derive from label
                derived = self._derive_canonical_from_label(obs.pattern_label)
                if derived:
                    obs.canonical_value = derived
                    obs.canonical_source = "pattern_label"
            elif result.matched and obs.entity:
                # Entity-only — no canonical value (not a problem signal)
                pass

        # ── 5. Flag generic sentiment phrases (don't set canonical_value) ──
        if obs.canonical_value and value_lower in (
            g.lower() for g in self._generic_sentiment
        ):
            # Generic phrase matched — clear canonical value so
            # ProblemSignalDiscoverer can filter these out.
            # Exception: if also matched a pattern label, keep canonical.
            if not obs.pattern_label:
                obs.canonical_value = None
                obs.canonical_source = None

        elapsed = time.time() - start

        if self.debug:
            debug_result = DebugResolutionResult(
                observation_id=result.observation_id,
                original_value=result.original_value,
                matched=result.matched,
                method=result.method,
                confidence=result.confidence,
                resolved_entity=result.resolved_entity,
                resolved_type=result.resolved_type,
                resolved_category=result.resolved_category,
                aliases_checked=list(self._entity_aliases.keys())[:10],
                candidate_entities=[(n, 0.5) for n in list(self._entity_names)[:5]],
                normalization_applied=(
                    f"{obs.value} -> {obs.canonical_value}" if obs.canonical_value else None
                ),
                resolution_time_ms=round(elapsed * 1000, 2),
            )
            return debug_result

        return result

    def enrich_batch(self, observations: list[Observation]) -> list[ResolutionResult]:
        """Batch enrich observations. Returns resolution metadata."""
        return [self.enrich(obs) for obs in observations]

    @staticmethod
    def _word_boundary_match(keyword: str, text: str) -> bool:
        """Check if keyword appears as a word or phrase boundary in text."""
        idx = text.find(keyword)
        if idx == -1:
            return False
        # Check word boundary before
        if idx > 0 and text[idx - 1].isalnum():
            return False
        # Check word boundary after
        end = idx + len(keyword)
        if end < len(text) and text[end].isalnum():
            return False
        return True

    @staticmethod
    def _derive_canonical_from_label(label: str) -> str | None:
        """Derive a readable canonical value from a pattern label."""
        # Convert snake_case to Title Case
        parts = label.replace("-", "_").split("_")
        if not parts:
            return None
        return " ".join(p.capitalize() for p in parts)