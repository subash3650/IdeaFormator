"""N-gram extraction: bigrams, trigrams, and fourgrams."""

from __future__ import annotations

from collections import Counter

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


class NgramExtractor(Extractor):
    """Extracts n-gram observations from document text."""

    def __init__(
        self,
        min_frequency: int = 5,
        max_features: int = 100,
        confidence: ConfidencePolicy | None = None,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.min_frequency = min_frequency
        self.max_features = max_features
        self.confidence = confidence or ConfidencePolicy()
        self.pipeline_version = pipeline_version

    @property
    def name(self) -> str:
        return "ngrams"

    def extract(self, df: pl.DataFrame) -> ObservationBundle:
        obs_list: list[Observation] = []
        text_col = self._resolve_text_column(df)

        bigram_c: Counter[str] = Counter()
        trigram_c: Counter[str] = Counter()
        fourgram_c: Counter[str] = Counter()

        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            if not text:
                continue
            tokens = text.lower().split()
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                bigram_c[" ".join(tokens[i:i+2])] += 1
            if len(tokens) >= 3:
                for i in range(len(tokens) - 2):
                    trigram_c[" ".join(tokens[i:i+3])] += 1
            if len(tokens) >= 4:
                for i in range(len(tokens) - 3):
                    fourgram_c[" ".join(tokens[i:i+4])] += 1

        for grams, otype in [(bigram_c, ObservationType.BIGRAM),
                              (trigram_c, ObservationType.TRIGRAM),
                              (fourgram_c, ObservationType.FOURGRAM)]:
            filtered = {k: v for k, v in grams.items() if v >= self.min_frequency}
            top = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
            for text, count in top:
                ctx = ExtractionContext(method=ExtractionMethod.STATISTICAL, frequency=count)
                obs_list.append(Observation(
                    observation_id=observation_id(otype, text, "global"),
                    type=otype, value=text, document_id="global",
                    platform="", extractor=self.name,
                    method=ExtractionMethod.STATISTICAL,
                    confidence=self.confidence.for_extraction(ctx),
                    pipeline_version=self.pipeline_version, generated_at="",
                ))

        return ObservationBundle(extractor=self.name, observations=obs_list)

    @staticmethod
    def _resolve_text_column(df: pl.DataFrame) -> str:
        for col in ("clean_text", "text"):
            if col in df.columns:
                return col
        return df.columns[0]