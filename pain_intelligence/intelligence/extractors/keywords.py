"""Keyword extraction using TF-IDF and RAKE."""

from __future__ import annotations

import re
from collections import Counter
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

_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "not", "no", "nor", "so", "very", "just", "also", "too", "as", "if",
    "then", "than", "because", "about", "into", "over", "after", "before",
    "between", "through", "during", "up", "down", "out", "off", "under",
    "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "what", "which", "who", "whom",
}


class KeywordExtractor(Extractor):
    """Extracts keywords using TF-IDF (Polars-native) and RAKE."""

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
        return "keywords"

    def extract(self, df: pl.DataFrame) -> ObservationBundle:
        obs_list: list[Observation] = []
        text_col = self._resolve_text_column(df)

        # TF-IDF via Polars
        tfidf_obs = self._extract_tfidf(df, text_col)
        obs_list.extend(tfidf_obs)

        # RAKE
        rake_obs = self._extract_rake(df, text_col)
        obs_list.extend(rake_obs)

        return ObservationBundle(extractor=self.name, observations=obs_list)

    def _extract_tfidf(self, df: pl.DataFrame, text_col: str) -> list[Observation]:
        obs: list[Observation] = []
        total_docs = df.height
        if total_docs == 0:
            return obs

        # Tokenize and count term frequencies per document
        tf_data: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            doc_id: str = str(row.get("id", ""))
            tokens = [t for t in text.lower().split() if len(t) > 2 and t not in _STOPWORDS]
            if not tokens:
                continue
            counter = Counter(tokens)
            for term, count in counter.most_common(20):
                tf_data.append({"doc_id": doc_id, "term": term, "tf": count})

        if not tf_data:
            return obs

        tf_df = pl.DataFrame(tf_data)

        # Document frequency per term
        df_counts = tf_df.group_by("term").agg(
            pl.col("doc_id").n_unique().alias("doc_freq"),
            pl.col("tf").sum().alias("total_tf"),
        ).filter(pl.col("doc_freq") >= 2)

        # Compute TF-IDF
        df_counts = df_counts.with_columns(
            (pl.col("total_tf") * ((pl.lit(total_docs + 1) / (pl.col("doc_freq") + 1)).log() + 1)).alias("tfidf")
        )

        top = df_counts.sort("tfidf", descending=True).head(self.max_features)

        for row in top.iter_rows(named=True):
            ctx = ExtractionContext(method=ExtractionMethod.STATISTICAL, frequency=int(row["total_tf"]))
            obs.append(Observation(
                observation_id=observation_id(ObservationType.KEYWORD_TFIDF, row["term"], "global"),
                type=ObservationType.KEYWORD_TFIDF,
                value=row["term"],
                document_id="global",
                platform="", rating=None, country=None,
                extractor=self.name, method=ExtractionMethod.STATISTICAL,
                confidence=self.confidence.for_extraction(ctx),
                pipeline_version=self.pipeline_version, generated_at="",
            ))

        return obs

    def _extract_rake(self, df: pl.DataFrame, text_col: str) -> list[Observation]:
        """RAKE (Rapid Automatic Keyword Extraction) implementation."""
        obs: list[Observation] = []
        phrase_scores: Counter[str] = Counter()
        word_scores: dict[str, float] = {}
        word_freq: Counter[str] = Counter()
        word_degree: Counter[str] = Counter()

        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())

            # Split on stopwords to get candidate phrases
            phrases: list[list[str]] = []
            current: list[str] = []
            for t in tokens:
                if t in _STOPWORDS or len(t) <= 2:
                    if current:
                        phrases.append(current)
                        current = []
                else:
                    current.append(t)
                    word_freq[t] += 1
                    word_degree[t] += 1
            if current:
                phrases.append(current)

            for phrase in phrases:
                for w in phrase:
                    word_degree[w] += len(phrase) - 1

        # Compute word scores: degree / frequency
        for word, freq in word_freq.items():
            deg = word_degree.get(word, 1)
            word_scores[word] = deg / freq

        # Score phrases
        for row in df.iter_rows(named=True):
            text: str = row.get(text_col, "") or ""
            tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
            current: list[str] = []
            for t in tokens:
                if t in _STOPWORDS or len(t) <= 2:
                    if current:
                        phrase = " ".join(current)
                        score = sum(word_scores.get(w, 0) for w in current)
                        phrase_scores[phrase] += score
                        current = []
                else:
                    current.append(t)
            if current:
                phrase = " ".join(current)
                score = sum(word_scores.get(w, 0) for w in current)
                phrase_scores[phrase] += score

        top = phrase_scores.most_common(self.max_features)
        for phrase, score in top:
            ctx = ExtractionContext(method=ExtractionMethod.HEURISTIC, frequency=int(score))
            obs.append(Observation(
                observation_id=observation_id(ObservationType.KEYWORD_RAKE, phrase, "global"),
                type=ObservationType.KEYWORD_RAKE,
                value=phrase,
                document_id="global",
                platform="", rating=None, country=None,
                extractor=self.name, method=ExtractionMethod.HEURISTIC,
                confidence=self.confidence.for_extraction(ctx),
                pipeline_version=self.pipeline_version, generated_at="",
            ))

        return obs

    @staticmethod
    def _resolve_text_column(df: pl.DataFrame) -> str:
        for col in ("clean_text", "text"):
            if col in df.columns:
                return col
        return df.columns[0]