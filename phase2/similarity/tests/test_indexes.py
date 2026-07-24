"""Tests for vector indexes.

Covers LinearIndex operations and VectorIndex ABC future compatibility."""
from __future__ import annotations

import numpy as np
import pytest

from phase2.similarity.indexes import LinearIndex, VectorIndex


class TestLinearIndex:
    def test_basic_search(self) -> None:
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        idx = LinearIndex(vectors)
        scores, indices = idx.search(np.array([1, 0, 0], dtype=np.float32), k=2)
        assert len(scores) == 2
        assert len(indices) == 2
        assert indices[0] == 0
        assert scores[0] == pytest.approx(1.0)

    def test_search_batch(self) -> None:
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        idx = LinearIndex(vectors)
        queries = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        scores, indices = idx.search_batch(queries, k=2)
        assert scores.shape == (2, 2)
        assert indices.shape == (2, 2)
        assert indices[0, 0] == 0
        assert indices[1, 0] == 1

    def test_size_and_dimension(self) -> None:
        vectors = np.random.randn(100, 384).astype(np.float32)
        idx = LinearIndex(vectors)
        assert idx.size == 100
        assert idx.dimension == 384

    def test_empty_index(self) -> None:
        vectors = np.zeros((0, 10), dtype=np.float32)
        idx = LinearIndex(vectors)
        assert idx.size == 0

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="Expected 2D"):
            LinearIndex(np.array([1, 2, 3], dtype=np.float32))

    def test_top_k_ordering(self) -> None:
        vectors = np.array([
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0.8, 0.2, 0],
        ], dtype=np.float32)
        idx = LinearIndex(vectors)
        scores, indices = idx.search(np.array([1, 0, 0], dtype=np.float32), k=3)
        assert scores[0] >= scores[1] >= scores[2]
        assert indices[0] == 0

    def test_k_larger_than_index(self) -> None:
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        idx = LinearIndex(vectors)
        scores, indices = idx.search(np.array([1, 0], dtype=np.float32), k=100)
        assert len(scores) == 2
        assert len(indices) == 2


class TestVectorIndexFutureCompatibility:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            VectorIndex()

    def test_can_subclass(self) -> None:
        class FakeIndex(VectorIndex):
            def search(self, query, k):
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
            def search_batch(self, queries, k):
                return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64)
            @property
            def size(self) -> int:
                return 0
            @property
            def dimension(self) -> int:
                return 0
        idx = FakeIndex()
        assert idx.size == 0
        assert idx.dimension == 0

    def test_save_not_implemented(self) -> None:
        class MinimalIndex(VectorIndex):
            def search(self, query, k):
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
            def search_batch(self, queries, k):
                return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64)
            @property
            def size(self) -> int:
                return 0
            @property
            def dimension(self) -> int:
                return 0
        idx = MinimalIndex()
        with pytest.raises(NotImplementedError):
            idx.save("/tmp/test.idx")

    def test_load_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            VectorIndex.load("/tmp/test.idx")

    def test_linear_index_is_vectorindex(self) -> None:
        vectors = np.random.randn(10, 64).astype(np.float32)
        idx = LinearIndex(vectors)
        assert isinstance(idx, VectorIndex)

    def test_abc_method_signatures(self) -> None:
        import inspect
        sig = inspect.signature(VectorIndex.search)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "k" in params
        sig2 = inspect.signature(VectorIndex.search_batch)
        params2 = list(sig2.parameters.keys())
        assert "queries" in params2
        assert "k" in params2