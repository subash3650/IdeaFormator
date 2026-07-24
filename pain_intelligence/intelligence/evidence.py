"""Evidence construction: aggregates observations into structured evidence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from pain_intelligence.intelligence.confidence import ConfidencePolicy, EvidenceStats
from pain_intelligence.intelligence.schema import (
    EntityType,
    Evidence,
    evidence_id,
)


class AggregationStrategy(ABC):
    """Pluggable strategy for aggregating observations into evidence."""

    @abstractmethod
    def aggregate(self, observations: list[Any]) -> list[Evidence]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class RuleAggregation(AggregationStrategy):
    """Default rule-based aggregation.

    Groups observations by (entity, category, canonical_value).
    Uses canonical_value when available to merge semantic variants into one evidence record.
    Computes aggregate statistics: count, avg_rating, distributions.
    """

    def __init__(self, confidence: ConfidencePolicy | None = None) -> None:
        self.confidence = confidence or ConfidencePolicy()

    @property
    def name(self) -> str:
        return "rule"

    def aggregate(self, observations: list[Any]) -> list[Evidence]:
        from pain_intelligence.intelligence.schema import Observation

        groups: dict[str, list[Observation]] = {}

        for obs in observations:
            if not isinstance(obs, Observation):
                continue
            # Build signal key from entity + category + canonical_value (or raw value)
            entity_part = obs.entity or ""
            cat_part = obs.category or ""
            value_part = (obs.canonical_value or obs.value)[:60]
            key = f"{cat_part}:{value_part}:{entity_part}".strip(":").lower()
            if not key:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(obs)

        evidence_list: list[Evidence] = []

        for signal_key, obs_group in groups.items():
            doc_ids: set[str] = set()
            platforms: Counter[str] = Counter()
            countries: Counter[str] = Counter()
            ratings: list[float] = []
            snippets: list[str] = []
            total_conf = 0.0
            entities: set[str] = set()
            categories: set[str] = set()

            for o in obs_group:
                doc_ids.add(o.document_id)
                if o.platform:
                    platforms[o.platform] += 1
                if o.country:
                    countries[o.country] += 1
                if o.rating is not None:
                    ratings.append(o.rating)
                if o.text_snippet:
                    snippets.append(o.text_snippet)
                total_conf += o.confidence
                if o.entity:
                    entities.add(o.entity)
                if o.category:
                    categories.add(o.category)

            if len(obs_group) < 3:
                continue

            avg_rating = sum(ratings) / len(ratings) if ratings else None
            avg_conf = total_conf / len(obs_group) if obs_group else 0.0

            # Pick top snippets
            top_snippets = sorted(snippets, key=len, reverse=True)[:5]

            stats = EvidenceStats(
                observation_count=len(obs_group),
                document_count=len(doc_ids),
                rating_std=self._compute_std(ratings) if ratings else 0.0,
                platform_diversity=len(platforms),
                country_diversity=len(countries),
                avg_observation_confidence=avg_conf,
            )

            evidence_list.append(Evidence(
                evidence_id=evidence_id(signal_key, next(iter(entities), "")),
                signal_key=signal_key,
                category=next(iter(categories), None),
                entity=next(iter(entities), None),
                entity_type=EntityType.UNKNOWN,
                signal_text=obs_group[0].canonical_value or obs_group[0].value[:60] if obs_group else "",
                observation_count=len(obs_group),
                document_count=len(doc_ids),
                avg_rating=avg_rating,
                platform_distribution=dict(platforms.most_common()),
                country_distribution=dict(countries.most_common()),
                observation_ids=[o.observation_id for o in obs_group[:100]],
                top_snippets=top_snippets,
                confidence=self.confidence.for_evidence(stats),
                aggregation_strategy=self.name,
                pipeline_version="1.5.0",
                generated_at=datetime.now(timezone.utc).isoformat(),
            ))

        return evidence_list

    @staticmethod
    def _compute_std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5


class EvidenceBuilder:
    """Builds evidence using a pluggable aggregation strategy.
    
    Default: RuleAggregation. Extensible via strategy.setter or subclass.
    """

    def __init__(self, strategy: AggregationStrategy | None = None) -> None:
        self._strategy: AggregationStrategy = strategy or RuleAggregation()

    @property
    def strategy(self) -> AggregationStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AggregationStrategy) -> None:
        self._strategy = strategy

    def build(self, observations: list[Any]) -> list[Evidence]:
        """Build evidence from observations using the current strategy."""
        return self._strategy.aggregate(observations)