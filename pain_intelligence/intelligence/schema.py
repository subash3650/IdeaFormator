"""Pydantic models and enums for the Knowledge Extraction Engine.

Observation → Knowledge Enrichment → Evidence → Problem Signals.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────

class ObservationType(str, Enum):
    ENTITY = "entity"
    PHRASE = "phrase"
    BIGRAM = "bigram"
    TRIGRAM = "trigram"
    FOURGRAM = "fourgram"
    KEYWORD_TFIDF = "keyword_tfidf"
    KEYWORD_RAKE = "keyword_rake"
    PATTERN_MATCH = "pattern_match"


class ExtractionMethod(str, Enum):
    DICTIONARY_MATCH = "dictionary_match"
    PATTERN_MATCH = "pattern_match"
    STATISTICAL = "statistical"
    HEURISTIC = "heuristic"


class EntityType(str, Enum):
    COMPANY = "company"
    PRODUCT = "product"
    PAYMENT_METHOD = "payment_method"
    LOGISTICS = "logistics"
    TECHNOLOGY = "technology"
    SERVICE = "service"
    LOCATION = "location"
    UNKNOWN = "unknown"


class ConfidenceSource(str, Enum):
    EXACT_MATCH = "exact_match"
    PATTERN_MATCH = "pattern_match"
    FUZZY_MATCH = "fuzzy_match"
    FREQUENCY_BASED = "frequency_based"
    STATISTICAL = "statistical"


# ── Deterministic ID helpers ──────────────────────────────────────

def _compute_id(prefix: str, *parts: str) -> str:
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def observation_id(obs_type: ObservationType, value: str, doc_id: str) -> str:
    return _compute_id("obs", obs_type.value, value, doc_id)


def evidence_id(signal_key: str, entity: str = "") -> str:
    return _compute_id("ev", signal_key, entity)


def signal_id(signal_key: str, entity: str = "", country: str = "") -> str:
    return _compute_id("sig", signal_key, entity, country)


# ── Core Models ───────────────────────────────────────────────────

class Observation(BaseModel):
    """A single raw fact extracted from a document.
    
    No enrichment. No inference. Just extraction.
    Knowledge enrichment mutates entity/entity_type/category in-place.
    """
    observation_id: str
    type: ObservationType
    value: str
    document_id: str
    platform: str
    rating: float | None = None
    country: str | None = None
    text_snippet: str = ""
    extractor: str = ""
    method: ExtractionMethod = ExtractionMethod.HEURISTIC
    confidence: float = 0.0

    # Knowledge enrichment (nullable before resolution)
    entity: str | None = None
    entity_type: EntityType | None = None
    category: str | None = None

    # Pattern label (set by PatternMatcher for pattern-matched observations)
    pattern_label: str | None = None

    # Canonicalization (set by KnowledgeEnricher)
    # canonical_value: the resolved canonical signal concept (e.g., "Late Delivery")
    # canonical_source: how canonicalization was derived (e.g., "taxonomy", "problem_signals", "pattern", "alias")
    canonical_value: str | None = None
    canonical_source: str | None = None

    pipeline_version: str = ""
    generated_at: str = ""


class ObservationBundle(BaseModel):
    """Collection of observations from a single extractor."""
    extractor: str
    observations: list[Observation] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for obs in self.observations:
            c[obs.type.value] = c.get(obs.type.value, 0) + 1
        return c

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataframe(self) -> "pl.DataFrame":  # noqa: F821
        import polars as pl
        records = [o.model_dump() for o in self.observations]
        if not records:
            return pl.DataFrame()
        # Convert enums to strings for Polars
        for r in records:
            r["type"] = r["type"].value if r["type"] else None
            r["method"] = r["method"].value if r["method"] else None
            r["entity_type"] = r["entity_type"].value if r["entity_type"] else None
        return pl.DataFrame(records)

    @classmethod
    def merge(cls, bundles: list[ObservationBundle]) -> list[Observation]:
        """Merge multiple bundles into a single flat list."""
        result: list[Observation] = []
        for b in bundles:
            result.extend(b.observations)
        return result


class Evidence(BaseModel):
    """Aggregated conclusion from multiple observations.
    
    Evidence is a statistical conclusion:
    "12K docs contain 'late delivery' + Amazon + rating ≤ 2."
    """
    evidence_id: str
    signal_key: str
    category: str | None = None
    entity: str | None = None
    entity_type: EntityType | None = None
    signal_text: str = ""

    observation_count: int = 0
    document_count: int = 0
    avg_rating: float | None = None
    platform_distribution: dict[str, int] = Field(default_factory=dict)
    country_distribution: dict[str, int] = Field(default_factory=dict)

    observation_ids: list[str] = Field(default_factory=list)
    top_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    aggregation_strategy: str = "rule"

    pipeline_version: str = ""
    generated_at: str = ""

    @staticmethod
    def to_dataframe(evidences: list[Evidence]) -> "pl.DataFrame":  # noqa: F821
        import json
        import polars as pl
        records = []
        for e in evidences:
            d = e.model_dump()
            if d.get("entity_type"):
                d["entity_type"] = d["entity_type"].value
            for key, val in d.items():
                if isinstance(val, dict):
                    d[key] = json.dumps(val)
            records.append(d)
        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)


class ProblemSignal(BaseModel):
    """A validated evidence cluster worth investigating.
    
    NOT an opportunity signal. NOT scored.
    A well-supported pattern that warrants human attention.
    """
    signal_key: str
    category: str | None = None
    entity: str | None = None
    entity_type: EntityType | None = None
    country: str | None = None
    signal_text: str = ""

    document_count: int = 0
    avg_rating: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    observation_count: int = 0
    confidence: float = 0.0

    pipeline_version: str = ""
    generated_at: str = ""

    @staticmethod
    def to_dataframe(signals: list[ProblemSignal]) -> "pl.DataFrame":  # noqa: F821
        import polars as pl
        records = []
        for s in signals:
            d = s.model_dump()
            if d.get("entity_type"):
                d["entity_type"] = d["entity_type"].value
            records.append(d)
        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)


class ResolutionResult(BaseModel):
    """Knowledge enrichment result metadata.
    
    Production: matched, method, confidence only.
    Debug: full aliases_checked + candidate_entities (via DebugResolutionResult).
    """
    observation_id: str
    original_value: str
    matched: bool = False
    method: str = ""
    confidence: float = 0.0
    resolved_entity: str | None = None
    resolved_type: EntityType | None = None
    resolved_category: str | None = None


class DebugResolutionResult(ResolutionResult):
    """Extended resolution result for debug mode."""
    aliases_checked: list[str] = []
    candidate_entities: list[tuple[str, float]] = []
    normalization_applied: str | None = None
    resolution_time_ms: float = 0.0