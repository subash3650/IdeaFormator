"""Tests for similarity providers."""

from __future__ import annotations

import numpy as np
import pytest

from phase2.similarity.providers.base import SimilarityProvider
from phase2.similarity.providers.cosine import CosineSimilarityProvider
from phase2.similarity.providers.dot_product import DotProductSimilarityProvider
from phase2.similarity.providers.euclidean import EuclideanSimilarityProvider
from phase2.similarity.providers.registry import (
    available_providers,
    create_provider,
    get_provider_class,
)
from phase2.similarity.config import SimilarityEngineConfig


class TestCosineProvider:
    def test_pairwise(self) -> None:
        p = CosineSimilarityProvider()
        m1 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        m2 = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
        result = p.compute_pairwise(m1, m2)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[1, 0] == pytest.approx(0.0)

    def test_scores(self) -> None:
        p = CosineSimilarityProvider()
        query = np.array([1, 0, 0], dtype=np.float32)
        index = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        scores = p.compute_scores(query, index)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_name_and_version(self) -> None:
        p = CosineSimilarityProvider()
        assert p.name == "cosine"
        assert p.version == "1.0"


class TestDotProductProvider:
    def test_pairwise(self) -> None:
        p = DotProductSimilarityProvider()
        m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        m2 = np.array([[1, 0], [0, 1]], dtype=np.float32)
        result = p.compute_pairwise(m1, m2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(2.0)
        assert result[1, 0] == pytest.approx(3.0)

    def test_name(self) -> None:
        p = DotProductSimilarityProvider()
        assert p.name == "dot_product"


class TestEuclideanProvider:
    def test_identical_vectors(self) -> None:
        p = EuclideanSimilarityProvider()
        v = np.array([[1, 0, 0]], dtype=np.float32)
        result = p.compute_pairwise(v, v)
        assert result[0, 0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        p = EuclideanSimilarityProvider()
        m1 = np.array([[1, 0]], dtype=np.float32)
        m2 = np.array([[0, 1]], dtype=np.float32)
        result = p.compute_pairwise(m1, m2)
        assert 0.0 < result[0, 0] < 1.0

    def test_name(self) -> None:
        p = EuclideanSimilarityProvider()
        assert p.name == "euclidean"


class TestRegistry:
    def test_all_registered(self) -> None:
        names = available_providers()
        assert "cosine" in names
        assert "dot_product" in names
        assert "euclidean" in names

    def test_get_provider_class(self) -> None:
        cls = get_provider_class("cosine")
        assert cls is CosineSimilarityProvider

    def test_unknown_provider(self) -> None:
        with pytest.raises(KeyError, match="Unknown provider"):
            get_provider_class("nonexistent")

    def test_create_provider(self) -> None:
        cfg = SimilarityEngineConfig(metric="cosine")
        p = create_provider(cfg)
        assert isinstance(p, CosineSimilarityProvider)
