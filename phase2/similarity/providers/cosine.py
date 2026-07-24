"""Cosine similarity provider."""

from __future__ import annotations

import numpy as np

from phase2.similarity.providers.base import SimilarityProvider
from phase2.similarity.providers.registry import register


@register("cosine")
class CosineSimilarityProvider(SimilarityProvider):
    """Exact cosine similarity via NumPy dot product.

    Assumes vectors are L2-normalized (as produced by the embedding engine).
    """

    def compute_pairwise(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        return matrix1 @ matrix2.T

    def compute_scores(self, query: np.ndarray, index: np.ndarray) -> np.ndarray:
        return index @ query

    @property
    def name(self) -> str:
        return "cosine"

    @property
    def version(self) -> str:
        return "1.0"
