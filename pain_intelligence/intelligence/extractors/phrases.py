"""Phrase extraction: action verbs, nouns, adjective-noun combinations."""

from __future__ import annotations

import re
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

# Simple heuristic POS patterns using suffix/prefix rules
_VERB_SUFFIXES = ("ed", "ing", "ate", "ize", "ify", "en")
_NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "er", "or", "ist", "ism", "ing")
_ADJECTIVE_SUFFIXES = ("able", "ible", "al", "ful", "less", "ous", "ive", "ic", "ical", "ent", "ant")


class PhraseExtractor(Extractor):
    """Extracts phrases: action verbs, nouns, adjective-noun combinations."""

    def __init__(
        self,
        max_features: int = 100,
        confidence: ConfidencePolicy | None = None,
        pipeline_version: str = "1.5.0",
    ) -> None:
        self.max_features = max_features
        self.confidence = confidence or ConfidencePolicy()
        self.pipeline_version = pipeline_version

    @property
    def name(self) -> str:
        return "phrases"

    def extract(self, df: pl.DataFrame) -> ObservationBundle:
        obs_list: list[Observation] = []
        text_col = self._resolve_text_column(df)

        verb_c: Counter[str] = Counter()
        noun_c: Counter[str] = Counter()
        adj_noun_c: Counter[str] = Counter()

        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            doc_id: str = str(row.get("id", ""))
            if not text or not doc_id:
                continue

            tokens = text.lower().split()
            # Tag words by simple heuristic
            tagged = []
            for t in tokens:
                clean = re.sub(r'[^a-z]', '', t)
                if len(clean) < 3:
                    tagged.append((clean, "other"))
                elif clean.endswith(_VERB_SUFFIXES) and len(clean) > 3:
                    tagged.append((clean, "verb"))
                elif clean.endswith(_NOUN_SUFFIXES) or (clean.endswith("s") and len(clean) > 3):
                    tagged.append((clean, "noun"))
                elif clean.endswith(_ADJECTIVE_SUFFIXES):
                    tagged.append((clean, "adj"))
                else:
                    tagged.append((clean, "other"))

            for word, pos in tagged:
                if pos == "verb":
                    verb_c[word] += 1
                elif pos == "noun":
                    noun_c[word] += 1

            for i in range(len(tagged) - 1):
                if tagged[i][1] == "adj" and tagged[i+1][1] == "noun":
                    adj_noun_c[f"{tagged[i][0]} {tagged[i+1][0]}"] += 1

        # Create observations
        verb_top = verb_c.most_common(self.max_features)
        noun_top = noun_c.most_common(self.max_features)
        adj_noun_top = adj_noun_c.most_common(self.max_features)

        for word, count in verb_top:
            ctx = ExtractionContext(method=ExtractionMethod.HEURISTIC, frequency=count)
            obs_list.append(Observation(
                observation_id=observation_id(ObservationType.PHRASE, f"verb:{word}", "global"),
                type=ObservationType.PHRASE, value=f"verb:{word}",
                document_id="global", platform="",
                extractor=self.name, method=ExtractionMethod.HEURISTIC,
                confidence=self.confidence.for_extraction(ctx),
                pipeline_version=self.pipeline_version, generated_at="",
            ))

        for word, count in noun_top:
            ctx = ExtractionContext(method=ExtractionMethod.HEURISTIC, frequency=count)
            obs_list.append(Observation(
                observation_id=observation_id(ObservationType.PHRASE, f"noun:{word}", "global"),
                type=ObservationType.PHRASE, value=f"noun:{word}",
                document_id="global", platform="",
                extractor=self.name, method=ExtractionMethod.HEURISTIC,
                confidence=self.confidence.for_extraction(ctx),
                pipeline_version=self.pipeline_version, generated_at="",
            ))

        for phrase, count in adj_noun_top:
            ctx = ExtractionContext(method=ExtractionMethod.HEURISTIC, frequency=count)
            obs_list.append(Observation(
                observation_id=observation_id(ObservationType.PHRASE, f"adj_noun:{phrase}", "global"),
                type=ObservationType.PHRASE, value=f"adj_noun:{phrase}",
                document_id="global", platform="",
                extractor=self.name, method=ExtractionMethod.HEURISTIC,
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