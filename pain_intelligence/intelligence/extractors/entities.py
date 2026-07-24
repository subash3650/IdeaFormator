"""Entity extraction using seed dictionaries + heuristic patterns."""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from pain_intelligence.intelligence.confidence import ConfidencePolicy, ExtractionContext
from pain_intelligence.intelligence.extractors.base import Extractor
from pain_intelligence.intelligence.schema import (
    EntityType,
    ExtractionMethod,
    Observation,
    ObservationBundle,
    ObservationType,
    observation_id,
)


class EntityExtractor(Extractor):
    """Extracts named entities using seed dictionaries and heuristic patterns.
    
    Uses multiple complementary techniques:
    1. Exact dictionary match (high confidence)
    2. Fuzzy/alias match (medium confidence)
    3. Heuristic patterns for unknown entities (lower confidence)
    """

    def __init__(
        self,
        seed_entities: list[dict[str, Any]] | None = None,
        confidence: ConfidencePolicy | None = None,
        min_mentions: int = 3,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.seed_entities = seed_entities or []
        self.confidence = confidence or ConfidencePolicy()
        self.min_mentions = min_mentions
        self.pipeline_version = pipeline_version
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Build fast lookup structures from seed entities."""
        self._exact: dict[str, tuple[str, EntityType, list[str]]] = {}
        self._aliases: dict[str, tuple[str, EntityType, list[str]]] = {}
        for ent in self.seed_entities:
            name = ent["name"].lower()
            etype = EntityType(ent["type"]) if ent.get("type") else EntityType.UNKNOWN
            aliases_list = [a.lower() for a in ent.get("aliases", [])]
            self._exact[name] = (name, etype, aliases_list)
            for alias in aliases_list:
                self._aliases[alias] = (name, etype, aliases_list)

    @property
    def name(self) -> str:
        return "entities"

    def extract(self, df: pl.DataFrame) -> ObservationBundle:
        obs_list: list[Observation] = []
        text_col = self._resolve_text_column(df)

        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            doc_id: str = str(row.get("id", ""))
            platform: str = str(row.get("platform", ""))
            rating: float | None = row.get("rating")
            country: str | None = row.get("country")

            if not text or not doc_id:
                continue

            found = self._extract_from_text(text.lower(), doc_id, platform, rating, country)
            obs_list.extend(found)

        return ObservationBundle(extractor=self.name, observations=obs_list)

    def _extract_from_text(
        self,
        text: str,
        doc_id: str,
        platform: str,
        rating: float | None,
        country: str | None,
    ) -> list[Observation]:
        obs: list[Observation] = []
        seen: set[str] = set()

        # 1. Exact dictionary matches (high confidence)
        for name, (canonical, etype, _aliases) in self._exact.items():
            if name in text:
                key = f"entity:{canonical}:{doc_id}"
                if key in seen:
                    continue
                seen.add(key)
                ctx = ExtractionContext(
                    method=ExtractionMethod.DICTIONARY_MATCH,
                    exact_match=True,
                )
                obs.append(Observation(
                    observation_id=observation_id(ObservationType.ENTITY, canonical, doc_id),
                    type=ObservationType.ENTITY,
                    value=canonical,
                    document_id=doc_id,
                    platform=platform,
                    rating=rating,
                    country=country,
                    text_snippet=self._extract_snippet(text, canonical),
                    extractor=self.name,
                    method=ExtractionMethod.DICTIONARY_MATCH,
                    confidence=self.confidence.for_extraction(ctx),
                    pipeline_version=self.pipeline_version,
                    generated_at=__import__("datetime").datetime.now().isoformat(),
                ))

        # 2. Alias matches
        for alias, (canonical, etype, _aliases) in self._aliases.items():
            if alias in text and canonical not in seen:
                key = f"entity:{canonical}:{doc_id}"
                if key in seen:
                    continue
                seen.add(key)
                ctx = ExtractionContext(
                    method=ExtractionMethod.DICTIONARY_MATCH,
                    exact_match=False,
                )
                obs.append(Observation(
                    observation_id=observation_id(ObservationType.ENTITY, canonical, doc_id),
                    type=ObservationType.ENTITY,
                    value=canonical,
                    document_id=doc_id,
                    platform=platform,
                    rating=rating,
                    country=country,
                    text_snippet=self._extract_snippet(text, alias),
                    extractor=self.name,
                    method=ExtractionMethod.DICTIONARY_MATCH,
                    confidence=self.confidence.for_extraction(ctx),
                    pipeline_version=self.pipeline_version,
                    generated_at="",
                ))

        # 3. Heuristic: capitalized multi-word patterns as potential entities
        heuristic_matches = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', text)
        for match in set(heuristic_matches):
            ml = match.lower()
            if ml in self._exact or ml in self._aliases:
                continue
            key = f"entity_heuristic:{ml}:{doc_id}"
            if key in seen:
                continue
            seen.add(key)
            ctx = ExtractionContext(method=ExtractionMethod.HEURISTIC)
            obs.append(Observation(
                observation_id=observation_id(ObservationType.ENTITY, ml, doc_id),
                type=ObservationType.ENTITY,
                value=match,
                document_id=doc_id,
                platform=platform,
                rating=rating,
                country=country,
                text_snippet=self._extract_snippet(text, match),
                extractor=self.name,
                method=ExtractionMethod.HEURISTIC,
                confidence=self.confidence.for_extraction(ctx),
                pipeline_version=self.pipeline_version,
                generated_at="",
            ))

        return obs

    def _extract_snippet(self, text: str, term: str, window: int = 40) -> str:
        idx = text.lower().find(term.lower())
        if idx == -1:
            return text[:80]
        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    @staticmethod
    def _resolve_text_column(df: pl.DataFrame) -> str:
        for col in ("clean_text", "text"):
            if col in df.columns:
                return col
        return df.columns[0]