"""Tests for high-performance similarity computation."""

from __future__ import annotations

import numpy as np
import pytest

from phase2.similarity.comparer import count_frequencies, top_k_similarity
from phase2.similarity.providers.cosine import CosineSimilarityProvider


class TestTopKSimilarity:
    def test_basic(self) -> None:
        provider = CosineSimilarityProvider()
        queries = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        index = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        scores, indices = top_k_similarity(queries, index, provider, k=2)
        assert scores.shape == (2, 2)
        assert indices.shape == (2, 2)
        assert indices[0, 0] == 0
        assert indices[1, 0] == 1

    def test_k_larger_than_index(self) -> None:
        provider = CosineSimilarityProvider()
        queries = np.array([[1, 0]], dtype=np.float32)
        index = np.array([[1, 0], [0, 1]], dtype=np.float32)
        scores, indices = top_k_similarity(queries, index, provider, k=100)
        assert scores.shape == (1, 2)

    def test_empty_index(self) -> None:
        provider = CosineSimilarityProvider()
        queries = np.array([[1, 0]], dtype=np.float32)
        index = np.zeros((0, 2), dtype=np.float32)
        scores, indices = top_k_similarity(queries, index, provider, k=5)
        assert scores.shape == (1, 0)


class TestCountFrequencies:
    def test_basic(self) -> None:
        result = count_frequencies(["a", "b", "a", "c", "b", "a"])
        assert result["a"] == 3
        assert result["b"] == 2
        assert result["c"] == 1

    def test_empty(self) -> None:
        result = count_frequencies([])
        assert result == {}
