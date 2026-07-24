"""Pattern matching using reusable linguistic patterns."""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from pain_intelligence.intelligence.confidence import ConfidencePolicy, ExtractionContext
from pain_intelligence.intelligence.extractors.base import Extractor
from pain_intelligence.intelligence.schema import (
    ExtractionMethod,
    Observation,
    ObservationBundle,
    ObservationType,
    observation_id,
)


class PatternMatcher(Extractor):
    """Matches text against reusable linguistic patterns from seeds."""

    def __init__(
        self,
        patterns: list[dict[str, Any]] | None = None,
        confidence: ConfidencePolicy | None = None,
        min_confidence: float = 0.5,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.patterns = patterns or []
        self.confidence = confidence or ConfidencePolicy()
        self.min_confidence = min_confidence
        self.pipeline_version = pipeline_version
        self._compiled: list[dict[str, Any]] = []
        self._compile()

    def _compile(self) -> None:
        for p in self.patterns:
            for raw_pattern in p.get("patterns", []):
                # Convert "didn.t" to "didn['']?t" for handling contractions
                escaped = re.escape(raw_pattern).replace(r"\.t", r"[.'’]?t")
                self._compiled.append({
                    "category": p.get("category", "unknown"),
                    "label": p.get("label", raw_pattern),
                    "pattern_str": raw_pattern,
                    "regex": re.compile(escaped, re.IGNORECASE),
                    "base_confidence": p.get("confidence", 0.8),
                })

    @property
    def name(self) -> str:
        return "patterns"

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

            for pattern in self._compiled:
                match = pattern["regex"].search(text)
                if match:
                    match_confidence = pattern["base_confidence"]
                    if match_confidence < self.min_confidence:
                        continue

                    ctx = ExtractionContext(method=ExtractionMethod.PATTERN_MATCH, exact_match=True)
                    obs_list.append(Observation(
                        observation_id=observation_id(
                            ObservationType.PATTERN_MATCH,
                            f"{pattern['label']}:{match.group(0)}",
                            doc_id,
                        ),
                        type=ObservationType.PATTERN_MATCH,
                        value=match.group(0),
                        document_id=doc_id,
                        platform=platform,
                        rating=rating,
                        country=country,
                        text_snippet=self._extract_snippet(text, match),
                        extractor=self.name,
                        method=ExtractionMethod.PATTERN_MATCH,
                        confidence=self.confidence.for_extraction(ctx) * match_confidence,
                        pattern_label=pattern["label"],
                        pipeline_version=self.pipeline_version,
                        generated_at="",
                    ))

        return ObservationBundle(extractor=self.name, observations=obs_list)

    @staticmethod
    def _extract_snippet(text: str, match: re.Match, window: int = 40) -> str:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
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